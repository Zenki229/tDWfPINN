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
from src.physics.dw_pde import DWForward
from src.physics.irregular_2d import IrregularHole2D, LShape2D
from src.data.sampler import IrregularHoleSampler, LShapeSampler, TimeSpaceSampler
from src.vis.plotter import PlotlyPlotter, PltPlotter

log = logging.getLogger(__name__)


PDE_REGISTRY = {
    "dw_forward": DWForward,
    "forward": DWForward,
    "irregular_hole": IrregularHole2D,
    "lshape": LShape2D,
}


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
        
        # Plotter
        plot_backend = "plotly"
        if hasattr(cfg, "plot") and hasattr(cfg.plot, "backend"):
            plot_backend = str(cfg.plot.backend).lower()
        if plot_backend == "matplotlib":
            self.plotter = PltPlotter(os.getcwd(), cfg)
        else:
            self.plotter = PlotlyPlotter(os.getcwd(), cfg)
        self._setup_timing()

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
                    ],
                )
                writer.writeheader()

    def _record_timing(self, step: int, loss_value: float):
        if not self.timing_enabled or step % self.timing_epoch_steps != 0:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        elapsed = now - self.timing_epoch_start
        total = now - self.timing_start
        self.timing_epochs.append(elapsed)
        row = {
            "epoch": len(self.timing_epochs),
            "step": step,
            "epoch_steps": self.timing_epoch_steps,
            "elapsed_seconds": f"{elapsed:.8f}",
            "total_seconds": f"{total:.8f}",
            "average_epoch_seconds": f"{float(np.mean(self.timing_epochs)):.8f}",
            "loss": f"{loss_value:.8e}",
        }
        with open(self.timing_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
        self.timing_epoch_start = now
        log.info(
            "Timing epoch %s at step %s: %.3f seconds",
            row["epoch"],
            step,
            elapsed,
        )
        
    def train(self):
        max_steps = self.train_cfg.max_steps
        pbar = tqdm(range(max_steps), desc="Training")
        self.timing_start = time.perf_counter()
        self.timing_epoch_start = self.timing_start
        
        for step in pbar:
            # 1. Sample
            points = self.sampler.sample()
            
            # 2. RAD Sampling Logic
            if self.train_cfg.rad.use and step > 0: # Maybe not every step? Original: every step.
                # Original logic:
                # node_rad = rad_sampler.sample()
                # residuals = pde.residual(net, node_rad)
                # rad_points = sampler.rad_sampler(residuals['in'], node_rad['in'], ...)
                # ... replace some points['in'] with rad_points
                
                # We'll implement a simplified version or full version
                rad_points_raw = self.rad_sampler.sample()
                
                # We need residual magnitude
                # Re-use pde residual calculation but we only need 'domain'
                # But pde.residual calculates all.
                # We can optimize pde to separate domain residual.
                # For now, call full residual
                res_dict = self.pde.residual(self.model, rad_points_raw)
                res_domain = res_dict['domain'] # Tensor
                
                n_rad = int(self.train_cfg.rad.ratio * self.train_cfg.batch_size.domain)
                rad_selected = self.sampler.rad_sampler(res_domain, rad_points_raw['domain'], n_rad)
                rad_selected = rad_selected.detach() # Detach to ensure leaf
                
                # Replace in current batch
                # points['domain'] is N x D
                # We replace last n_rad points or random?
                # Original: ind = np.random.choice...
                current_domain = points['domain']
                n_keep = current_domain.shape[0] - n_rad
                if n_keep > 0:
                    keep_idx = torch.randperm(current_domain.shape[0])[:n_keep]
                    points['domain'] = torch.cat([current_domain[keep_idx], rad_selected], dim=0)
                else:
                    points['domain'] = rad_selected

            # 3. Optimization Step
            def closure():
                self.optimizer.zero_grad()
                residuals = self.pde.residual(self.model, points)
                loss = 0
                log_dict = {}
                
                weights = self.cfg.pde.weights
                
                # Weighted sum
                if 'domain' in residuals:
                    l = torch.mean(torch.square(residuals['domain']))
                    loss += weights.domain * l
                    log_dict['loss_domain'] = l.item()
                    
                if 'boundary' in residuals:
                    l = torch.mean(torch.square(residuals['boundary']))
                    loss += weights.boundary * l
                    log_dict['loss_boundary'] = l.item()
                    
                if 'initial' in residuals:
                    l = torch.mean(torch.square(residuals['initial']))
                    loss += weights.initial * l
                    log_dict['loss_initial'] = l.item()
                    
                if 'initial_dt' in residuals:
                    l = torch.mean(torch.square(residuals['initial_dt']))
                    loss += weights.initial_dt * l
                    log_dict['loss_initial_dt'] = l.item()
                
                log_dict['loss_total'] = loss.item()
                
                loss.backward()
                
                # WandB logging inside closure? Usually outside.
                # But LBFGS calls closure multiple times.
                # We'll return loss and log outside or keep simple.
                # For Adam, closure is called once.
                return loss, log_dict

            loss, log_dict = closure()
            self.optimizer.step()
            self._record_timing(step + 1, log_dict['loss_total'])
            
            # Logging
            if step % 100 == 0:
                wandb.log(log_dict, step=step)
                pbar.set_postfix({'loss': log_dict['loss_total']})
            
            # Evaluation & Plotting
            if step % 1000 == 0 or step == max_steps - 1:
                self.evaluate(step)
                self.save_checkpoint(step)

        # LBFGS Phase
        if self.cfg.optimizer.lbfgs.use:
            log.info("Starting LBFGS...")
            self.lbfgs_optimizer = optim.LBFGS(
                self.model.parameters(),
                lr=self.cfg.optimizer.lbfgs.lr,
                max_iter=self.cfg.optimizer.lbfgs.max_iter,
                history_size=self.cfg.optimizer.lbfgs.history_size
            )
            
            # Resample for LBFGS? Or use fixed batch?
            # Usually fixed batch for LBFGS steps or resampling?
            # Original code re-samples or uses closure logic.
            # Original: loop over epochs, call step(closure).
            
            for i in range(self.cfg.optimizer.lbfgs.epochs):
                points = self.sampler.sample() # Resample per epoch
                def lbfgs_closure():
                    self.lbfgs_optimizer.zero_grad()
                    residuals = self.pde.residual(self.model, points)
                    loss = 0
                    weights = self.cfg.pde.weights
                    for key, val in residuals.items():
                        loss += weights.get(key, 1.0) * torch.mean(torch.square(val))
                    loss.backward()
                    return loss
                
                self.lbfgs_optimizer.step(lbfgs_closure)
                
                # Eval after LBFGS epoch
                self.evaluate(max_steps + i + 1)

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
            points_tensor = torch.from_numpy(points_np).float().to(self.device)
            
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
            wandb.log({"L2_Relative_Error": l2_error}, step=step)
            log.info(f"Step {step}: L2 Error = {l2_error:.2e}")

    def _predict_numpy(self, points_np: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(points_np), chunk_size):
                chunk = torch.from_numpy(points_np[start:start + chunk_size]).float().to(self.device)
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
                points_tensor = torch.from_numpy(points_np).float().to(self.device)
                true_grid = np.full_like(x_grid, np.nan, dtype=float)
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
            pred_grid = np.full_like(true_grid, np.nan, dtype=float)
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
        if rel_errors:
            mean_rel = float(np.nanmean(rel_errors))
            wandb.log({"L2_Relative_Error": mean_rel}, step=step)
            log.info(f"Step {step}: 2D mean L2 Error = {mean_rel:.2e}")

    def save_checkpoint(self, step):
        path = os.path.join(os.getcwd(), f"checkpoint_{step}.pt")
        torch.save(self.model.state_dict(), path)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    run:wandb.run = setup_wandb(cfg)
    trainer = Trainer(cfg, run)
    trainer.train()

if __name__ == "__main__":
    main()
