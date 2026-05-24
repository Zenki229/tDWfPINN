import os
if os.name == "nt":
    # Avoid duplicate OpenMP runtime aborts in some Windows Conda stacks.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
import numpy as np
import sys
import csv
import logging
import time
from pathlib import Path
from tqdm import tqdm
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.experiments import set_seed, setup_wandb
from src.physics.burgers import TimeFracBurgers1D
from src.physics.dw_pde import DWForward
from src.physics.irregular_2d import IrregularHole2D, LShape2D
from src.data.sampler import IrregularHoleSampler, LShapeSampler, TimeSpaceSampler
from src.vis.plotter import PlotlyPlotter, PltPlotter

log = logging.getLogger(__name__)


PDE_REGISTRY = {
    "dw_forward": DWForward,
    "forward": DWForward,
    "burgers": TimeFracBurgers1D,
    "irregular_hole": IrregularHole2D,
    "lshape": LShape2D,
}


def _cfg_get(config, name, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    try:
        return getattr(config, name)
    except (AttributeError, KeyError):
        return default


def _lbfgs_enabled(cfg: DictConfig) -> bool:
    optimizer_name = str(_cfg_get(cfg.optimizer, "name", "adam")).lower()
    lbfgs_cfg = _cfg_get(cfg.optimizer, "lbfgs")
    return (
        optimizer_name in {"lbfgs", "adam_lbfgs", "adam+lbfgs", "hybrid"}
        or bool(_cfg_get(lbfgs_cfg, "use", False))
    )


class Trainer:
    def __init__(self, cfg: DictConfig, run: wandb.run):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup
        set_seed(cfg.seed)
        self.train_cfg = cfg.training if hasattr(cfg, "training") else cfg.trainer
        
        if hasattr(cfg.pde, "input_dim"):
            cfg.model.input_dim = int(cfg.pde.input_dim)

        # Components
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        self.pde = self._build_pde()
        
        # Sampler
        # Convert list of lists to list of lists (Hydra ListConfig issue)
        batch_size = OmegaConf.to_container(self.train_cfg.batch_size, resolve=True)
        self.sampler = self._build_sampler(batch_size)
        
        if self.train_cfg.rad.use:
            rad_batch = OmegaConf.to_container(self.train_cfg.rad.batch, resolve=True)
            self.rad_sampler = self._build_sampler(rad_batch)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.optimizer.lr)
        self.use_hybrid_lbfgs = _lbfgs_enabled(cfg)
        self.lbfgs_optimizer = None
        self.loss_event = 0
        
        # Plotter
        plot_backend = "plotly"
        if hasattr(cfg, "plot") and hasattr(cfg.plot, "backend"):
            plot_backend = str(cfg.plot.backend).lower()
        if plot_backend == "matplotlib":
            self.plotter = PltPlotter(os.getcwd(), cfg)
        else:
            self.plotter = PlotlyPlotter(os.getcwd(), cfg)
        self._setup_timing()
        self.loss_log_every_steps = int(
            _cfg_get(self.train_cfg, "loss_log_every_steps", 100)
        )
        self.eval_every_steps = int(
            _cfg_get(self.train_cfg, "eval_every_steps", self.timing_epoch_steps)
        )
        self.eval_every_epochs = int(_cfg_get(self.train_cfg, "eval_every_epochs", 1))
        self._last_rad_kept_points = None
        self._last_rad_selected_points = None

    def _build_pde(self):
        pde_name = str(getattr(self.cfg.pde, "name", "dw_forward"))
        pde_class = PDE_REGISTRY.get(pde_name)
        if pde_class is None:
            raise ValueError(f"Unknown PDE name '{pde_name}'. Available: {sorted(PDE_REGISTRY)}")
        return pde_class(self.cfg, self.device)

    def _build_sampler(self, batch_size):
        pde_name = str(getattr(self.cfg.pde, "name", "dw_forward"))
        t_lim = list(self.cfg.pde.t_lim)
        if pde_name == "irregular_hole":
            return IrregularHoleSampler(
                time_lim=t_lim,
                batch_size=batch_size,
                device=self.device,
                center=self.cfg.pde.center,
                r0=self.cfg.pde.r0,
            )
        if pde_name == "lshape":
            return LShapeSampler(
                time_lim=t_lim,
                batch_size=batch_size,
                device=self.device,
            )
        x_lim = list(self.cfg.pde.x_lim)
        if not isinstance(x_lim[0], (list, tuple)):
            x_lim = [list(self.cfg.pde.x_lim)]
        return TimeSpaceSampler(
            spatial_lim=x_lim,
            time_lim=t_lim,
            device=self.device,
            batch_size=batch_size,
        )

    def _setup_timing(self):
        timing_cfg = getattr(self.train_cfg, "timing", None)
        self.timing_enabled = True if timing_cfg is None else bool(getattr(timing_cfg, "enabled", True))
        self.timing_epoch_steps = 5000 if timing_cfg is None else int(getattr(timing_cfg, "epoch_steps", 5000))
        self.timing_path = os.path.join(os.getcwd(), "timing.csv")
        self.timing_start = None
        self.timing_epoch_start = None
        self.timing_epochs = []
        self.timing_external_elapsed = False
        if self.timing_enabled:
            with open(self.timing_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "step",
                        "epoch_steps",
                        "elapsed_seconds",
                        "total_seconds",
                        "average_epoch_seconds",
                        "loss",
                        "adam_loss",
                        "lbfgs_loss",
                    ],
                )
                writer.writeheader()

    def _write_timing_record(
        self,
        step: int,
        elapsed: float,
        total: float,
        loss_value: float,
        adam_loss_value: float = None,
        lbfgs_loss_value: float = None,
    ):
        self.timing_epochs.append(elapsed)
        row = {
            "epoch": len(self.timing_epochs),
            "step": step,
            "epoch_steps": self.timing_epoch_steps,
            "elapsed_seconds": f"{elapsed:.8f}",
            "total_seconds": f"{total:.8f}",
            "average_epoch_seconds": f"{float(np.mean(self.timing_epochs)):.8f}",
            "loss": f"{loss_value:.8e}",
            "adam_loss": f"{adam_loss_value:.8e}" if adam_loss_value is not None else "",
            "lbfgs_loss": f"{lbfgs_loss_value:.8e}" if lbfgs_loss_value is not None else "",
        }
        with open(self.timing_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
        log.info(
            "Timing epoch %s at step %s: %.3f seconds",
            row["epoch"],
            step,
            elapsed,
        )
        timing_log = {
            "timing/epoch": int(row["epoch"]),
            "timing/elapsed_seconds": elapsed,
            "timing/average_epoch_seconds": float(np.mean(self.timing_epochs)),
            "timing/loss": loss_value,
            "train/adam_step": step,
        }
        if adam_loss_value is not None:
            timing_log["timing/adam_loss"] = adam_loss_value
        if lbfgs_loss_value is not None:
            timing_log["timing/lbfgs_loss"] = lbfgs_loss_value
        wandb.log(timing_log)

    def _record_timing(self, step: int, loss_value: float):
        if not self.timing_enabled or step % self.timing_epoch_steps != 0:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        elapsed = now - self.timing_epoch_start
        total = now - self.timing_start
        self.timing_epoch_start = now
        self._write_timing_record(step, elapsed, total, loss_value)

    def _record_timing_elapsed(
        self,
        step: int,
        elapsed: float,
        loss_value: float,
        adam_loss_value: float = None,
        lbfgs_loss_value: float = None,
    ):
        if not self.timing_enabled:
            return
        self.timing_external_elapsed = True
        total = float(np.sum(self.timing_epochs) + elapsed)
        self._write_timing_record(
            step,
            elapsed,
            total,
            loss_value,
            adam_loss_value=adam_loss_value,
            lbfgs_loss_value=lbfgs_loss_value,
        )

    def _sample_training_points(self, step: int, rad_on_first: bool = False):
        points = self.sampler.sample()
        if not self.train_cfg.rad.use:
            return points
        if step == 0 and not rad_on_first:
            return points

        rad_points_raw = self.rad_sampler.sample()
        res_dict = self.pde.residual(self.model, rad_points_raw)
        res_domain = res_dict["domain"]

        n_rad = int(self.train_cfg.rad.ratio * self.train_cfg.batch_size.domain)
        n_rad = min(max(n_rad, 1), points["domain"].shape[0])
        rad_selected = self.sampler.rad_sampler(
            res_domain,
            rad_points_raw["domain"],
            n_rad,
        ).detach()

        current_domain = points["domain"]
        n_keep = current_domain.shape[0] - n_rad
        if n_keep > 0:
            keep_idx = torch.randperm(current_domain.shape[0], device=self.device)[:n_keep]
            kept_points = current_domain[keep_idx]
            points["domain"] = torch.cat([kept_points, rad_selected], dim=0)
        else:
            kept_points = current_domain[:0]
            points["domain"] = rad_selected
        self._last_rad_kept_points = kept_points.detach().cpu().numpy()
        self._last_rad_selected_points = rad_selected.detach().cpu().numpy()
        return points

    def _loss_closure(self, points, optimizer=None, backward=True):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        residuals = self.pde.residual(self.model, points)
        weights = self.cfg.pde.weights
        loss = torch.zeros((), device=self.device)
        log_dict = {}

        for key, value in residuals.items():
            term = torch.mean(torch.square(value))
            loss = loss + float(_cfg_get(weights, key, 1.0)) * term
            log_dict[f"loss_{key}"] = float(term.detach().cpu().item())

        log_dict["loss_total"] = float(loss.detach().cpu().item())

        if backward:
            loss.backward()
        return loss, log_dict

    def _adam_step(self, points):
        loss, log_dict = self._loss_closure(
            points,
            optimizer=self.optimizer,
            backward=True,
        )
        self.optimizer.step()
        return float(loss.detach().cpu().item()), log_dict

    def _build_lbfgs_optimizer(self):
        lbfgs_cfg = self.cfg.optimizer.lbfgs
        return optim.LBFGS(
            self.model.parameters(),
            lr=float(lbfgs_cfg.lr),
            max_iter=int(lbfgs_cfg.max_iter),
            max_eval=int(lbfgs_cfg.max_eval) if _cfg_get(lbfgs_cfg, "max_eval") is not None else None,
            history_size=int(lbfgs_cfg.history_size),
        )

    def _lbfgs_step(self, points):
        if self.lbfgs_optimizer is None:
            self.lbfgs_optimizer = self._build_lbfgs_optimizer()

        def closure():
            loss, _ = self._loss_closure(
                points,
                optimizer=self.lbfgs_optimizer,
                backward=True,
            )
            return loss

        self.lbfgs_optimizer.step(closure)
        loss, log_dict = self._loss_closure(points, optimizer=None, backward=False)
        self.lbfgs_optimizer.zero_grad(set_to_none=True)
        return float(loss.detach().cpu().item()), log_dict

    def _log_loss_event(self, phase: str, loss_value: float, global_step: int, epoch: int):
        self.loss_event += 1
        payload = {
            "train/loss_event": self.loss_event,
            "train/loss_continuous": loss_value,
            "train/adam_step": global_step,
            "train/epoch": epoch,
            "train/phase_id": 0 if phase == "adam" else 1,
            "train/phase": phase,
        }
        if phase == "adam":
            payload["train/adam_loss"] = loss_value
        else:
            payload["train/lbfgs_loss"] = loss_value
            payload["train/lbfgs_max_iter"] = int(self.cfg.optimizer.lbfgs.max_iter)
        wandb.log(payload)

    def _steps_per_epoch(self):
        timing_cfg = getattr(self.train_cfg, "timing", None)
        return int(
            _cfg_get(
                self.train_cfg,
                "steps_per_epoch",
                _cfg_get(timing_cfg, "epoch_steps", self.train_cfg.max_steps),
            )
        )

    def _log_train_terms(self, log_dict, step):
        payload = dict(log_dict)
        payload["train/adam_step"] = step
        wandb.log(payload)

    def _should_log_loss(self, step: int, max_steps: int) -> bool:
        if self.loss_log_every_steps <= 0:
            return False
        return (
            step == 1
            or step == max_steps
            or step % self.loss_log_every_steps == 0
        )

    def _should_evaluate_step(self, step: int, max_steps: int) -> bool:
        if self.eval_every_steps <= 0:
            return step == max_steps
        return step == max_steps or step % self.eval_every_steps == 0

    def _should_evaluate_epoch(self, epoch: int, epochs: int) -> bool:
        if self.eval_every_epochs <= 0:
            return epoch == epochs
        return epoch == epochs or epoch % self.eval_every_epochs == 0
        
    def train(self):
        if self.use_hybrid_lbfgs:
            self._train_hybrid()
        else:
            self._train_adam_only()

    def _train_adam_only(self):
        max_steps = int(self.train_cfg.max_steps)
        pbar = tqdm(range(max_steps), desc="Training")
        self.timing_start = time.perf_counter()
        self.timing_epoch_start = self.timing_start
        
        for step_idx in pbar:
            step = step_idx + 1
            points = self._sample_training_points(step_idx, rad_on_first=False)
            _, log_dict = self._adam_step(points)
            self._record_timing(step, log_dict["loss_total"])
            
            if self._should_log_loss(step, max_steps):
                self._log_train_terms(log_dict, step)
                pbar.set_postfix({"loss": log_dict["loss_total"]})
            
            if self._should_evaluate_step(step, max_steps):
                self.evaluate(step)
                self.save_checkpoint(step)

    def _train_hybrid(self):
        max_steps = int(self.train_cfg.max_steps)
        steps_per_epoch = self._steps_per_epoch()
        if steps_per_epoch <= 0:
            raise ValueError("trainer.steps_per_epoch must be positive")
        epochs = max_steps // steps_per_epoch
        if epochs <= 0:
            raise ValueError("trainer.max_steps must be >= trainer.steps_per_epoch")
        if max_steps % steps_per_epoch != 0:
            ignored = max_steps - epochs * steps_per_epoch
            log.info("Ignoring %s trailing step(s) after epoch division", ignored)

        self.timing_epoch_steps = steps_per_epoch
        self.timing_start = time.perf_counter()
        self.timing_epoch_start = self.timing_start
        effective_steps = epochs * steps_per_epoch
        pbar = tqdm(total=effective_steps, desc="Hybrid Adam steps")
        global_step = 0

        for epoch_idx in range(epochs):
            epoch = epoch_idx + 1
            points = self._sample_training_points(global_step, rad_on_first=True)
            adam_loss = None
            adam_log = None
            adam_start = time.perf_counter()
            for _ in range(steps_per_epoch):
                adam_loss, adam_log = self._adam_step(points)
                global_step += 1
                pbar.update(1)
                if self._should_log_loss(global_step, effective_steps):
                    self._log_train_terms(adam_log, global_step)
                    self._log_loss_event("adam", adam_loss, global_step, epoch)
                    pbar.set_postfix({
                        "phase": "adam",
                        "step": global_step,
                        "loss": adam_loss,
                    })

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            adam_elapsed = time.perf_counter() - adam_start

            pbar.set_postfix({
                "phase": "lbfgs",
                "step": global_step,
                "adam": adam_loss,
            })
            lbfgs_loss, lbfgs_log = self._lbfgs_step(points)
            self._record_timing_elapsed(
                global_step,
                adam_elapsed,
                lbfgs_loss,
                adam_loss_value=adam_loss,
                lbfgs_loss_value=lbfgs_loss,
            )
            self._log_loss_event("lbfgs", lbfgs_loss, global_step, epoch)

            log_payload = dict(lbfgs_log)
            log_payload["train/adam_step"] = global_step
            log_payload["train/epoch"] = epoch
            log_payload["train/lbfgs_max_iter"] = int(self.cfg.optimizer.lbfgs.max_iter)
            wandb.log(log_payload)

            pbar.set_postfix({
                "phase": "lbfgs_done",
                "step": global_step,
                "adam": adam_loss,
                "lbfgs": lbfgs_loss,
            })
            if self._should_evaluate_epoch(epoch, epochs):
                self.evaluate(global_step)
                self.save_checkpoint(global_step)
        pbar.close()

    def evaluate(self, step):
        if getattr(self.pde, "spatial_dim", 1) == 2:
            self.evaluate_2d(step)
            return
        with torch.no_grad():
            # Create mesh
            # Assume 1D spatial for plotting
            t_eval = np.linspace(self.cfg.pde.t_lim[0], self.cfg.pde.t_lim[1], 100)
            x_eval = np.linspace(self.cfg.pde.x_lim[0], self.cfg.pde.x_lim[1], 100)
            T, X = np.meshgrid(t_eval, x_eval)
            
            points_np = np.stack([T.flatten(), X.flatten()], axis=1)
            points_tensor = torch.from_numpy(points_np).to(self.device)
            
            u_pred = self.model(points_tensor).cpu().numpy().reshape(T.shape)
            u_exact = self.pde.exact(points_tensor).cpu().numpy().reshape(T.shape)
            
            self.plotter.plot_solution(
                T,
                X,
                u_pred,
                f"Prediction step {step}",
                f"prediction_step_{step}"
            )
            
            # Log error metrics
            l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
            wandb.log({
                "train/adam_step": step,
                "eval/relative_error": l2_error,
                "L2_Relative_Error": l2_error,
            })
            log.info(f"Step {step}: L2 Error = {l2_error:.2e}")

    def _predict_numpy(self, points_np: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(points_np), chunk_size):
                chunk = torch.from_numpy(points_np[start:start + chunk_size]).to(self.device)
                outputs.append(self.model(chunk).detach().cpu().numpy().reshape(-1))
        return np.concatenate(outputs, axis=0)

    def evaluate_2d(self, step):
        time_values = self.pde.plot_time_slices()
        if hasattr(self.pde, "reference_slices"):
            x_grid, y_grid, selected = self.pde.reference_slices(time_values)
            slices = [(time_value, true_grid) for time_value, true_grid in selected]
        else:
            grid_n = int(getattr(self.cfg.pde, "plot_grid", 80))
            grid = np.linspace(-1.0, 1.0, grid_n)
            x_grid, y_grid = np.meshgrid(grid, grid, indexing="xy")
            mask = self.pde.domain_mask_np(x_grid, y_grid)
            slices = []
            for time_value in time_values:
                points_np = np.stack([
                    np.full(np.count_nonzero(mask), time_value),
                    x_grid[mask],
                    y_grid[mask],
                ], axis=1)
                points_tensor = torch.from_numpy(points_np).to(self.device)
                true_grid = np.full_like(x_grid, np.nan)
                true_grid[mask] = self.pde.exact(points_tensor).detach().cpu().numpy().reshape(-1)
                slices.append((time_value, true_grid))

        rel_errors = []
        for time_value, true_grid in slices:
            mask = np.isfinite(true_grid)
            points_np = np.stack([
                np.full(np.count_nonzero(mask), time_value),
                x_grid[mask],
                y_grid[mask],
            ], axis=1)
            pred_grid = np.full_like(true_grid, np.nan)
            pred_grid[mask] = self._predict_numpy(points_np)
            err_grid = np.abs(pred_grid - true_grid)
            denom = np.linalg.norm(true_grid[mask])
            rel_err = np.nan if denom < 1e-14 else np.linalg.norm(pred_grid[mask] - true_grid[mask]) / denom
            rel_errors.append(rel_err)
            token = f"t{time_value:.3f}".replace(".", "p")
            finite_vals = np.concatenate([true_grid[mask], pred_grid[mask]])
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))
            self.plotter.plot_2d_field(
                x_grid,
                y_grid,
                true_grid,
                f"true, t={time_value:.3f}",
                f"{self.cfg.pde.name}_{token}_true_step_{step}",
                vmin=vmin,
                vmax=vmax,
            )
            self.plotter.plot_2d_field(
                x_grid,
                y_grid,
                pred_grid,
                f"sol, t={time_value:.3f}",
                f"{self.cfg.pde.name}_{token}_sol_step_{step}",
                vmin=vmin,
                vmax=vmax,
            )
            self.plotter.plot_2d_field(
                x_grid,
                y_grid,
                err_grid,
                f"abs error, t={time_value:.3f}\nrelative error = {rel_err:.4e}",
                f"{self.cfg.pde.name}_{token}_abs_error_step_{step}",
                vmin=0.0,
                vmax=float(np.nanmax(err_grid)),
            )
            self._plot_rad_scatter_at_time(
                time_value, time_values, x_grid, y_grid, token, step
            )
        if rel_errors:
            mean_rel = float(np.nanmean(rel_errors))
            wandb.log({
                "train/adam_step": step,
                "eval/relative_error": mean_rel,
                "L2_Relative_Error": mean_rel,
            })
            log.info(f"Step {step}: 2D mean L2 Error = {mean_rel:.2e}")

    def _plot_rad_scatter_at_time(self, time_value, time_values, x_grid, y_grid, token, step):
        kept = self._last_rad_kept_points
        rad = self._last_rad_selected_points
        if kept is None and rad is None:
            return
        if kept is not None and kept.shape[1] < 3:
            return
        if rad is not None and rad.shape[1] < 3:
            return
        t_min = float(self.cfg.pde.t_lim[0])
        t_max = float(self.cfg.pde.t_lim[1])
        n_slices = max(len(time_values), 1)
        dt_tol = max((t_max - t_min) / (2 * n_slices), 1e-6)
        kept_xy = kept[np.abs(kept[:, 0] - time_value) <= dt_tol][:, 1:3] if kept is not None else None
        rad_xy = rad[np.abs(rad[:, 0] - time_value) <= dt_tol][:, 1:3] if rad is not None else None
        self.plotter.plot_2d_scatter_rad(
            kept_xy,
            rad_xy,
            f"RAD sampling, t={time_value:.3f}",
            f"{self.cfg.pde.name}_{token}_rad_scatter_step_{step}",
            xlim=(float(np.min(x_grid)), float(np.max(x_grid))),
            ylim=(float(np.min(y_grid)), float(np.max(y_grid))),
        )

    def save_checkpoint(self, step):
        path = os.path.join(os.getcwd(), f"checkpoint_{step}.pt")
        torch.save(self.model.state_dict(), path)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float64)
    log.info("Torch default dtype set to %s", torch.get_default_dtype())
    log.info(OmegaConf.to_yaml(cfg))
    run:wandb.run = setup_wandb(cfg)
    trainer = Trainer(cfg, run)
    trainer.train()

if __name__ == "__main__":
    main()
