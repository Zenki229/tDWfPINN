import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
import numpy as np
import os
import logging
from tqdm import tqdm
import time
import wandb
import sys 


from src.utils.experiments import set_seed, setup_wandb
from src.models.net import MLP
from src.physics.dw_pde import DWForward
from src.data.sampler import TimeSpaceSampler
from src.vis.plotter import PlotlyPlotter, PltPlotter

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg: DictConfig, run: wandb.run):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.run = run
        
        # Setup
        set_seed(cfg.seed)
        
        # Components
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        # self.pde = hydra.utils.instantiate(cfg.pde, device=self.device) 
        # Manually instantiate PDE because it expects the full config object
        pde_cls = hydra.utils.get_class(cfg.pde._target_)
        self.pde = pde_cls(cfg, device=self.device) 
        
        # Sampler
        # Convert list of lists to list of lists (Hydra ListConfig issue)
        x_lim = list(cfg.pde.x_lim)
        if not isinstance(x_lim[0], (list, tuple)):
            # Handle [0,1] vs [[0,1]] ambiguity if any
            # The config says x_lim: [0, 1]. The code expects [[0,1]] for 1D spatial.
            # Or [0,1] implies min/max for 1D?
            # Original config: xlim = [[0,1]].
            # My config: x_lim: [0, 1].
            # I should fix config or handle here.
            # Let's fix here to be safe.
            x_lim = [list(cfg.pde.x_lim)]
            
        t_lim = list(cfg.pde.t_lim)
        
        batch_size = OmegaConf.to_container(cfg.trainer.batch_size, resolve=True)
        self.sampler = TimeSpaceSampler(
            spatial_lim=x_lim,
            time_lim=t_lim,
            device=self.device,
            batch_size=batch_size
        )
        
        if cfg.trainer.rad.use:
            rad_batch = OmegaConf.to_container(cfg.trainer.rad.batch, resolve=True)
            self.rad_sampler = TimeSpaceSampler(
                spatial_lim=x_lim,
                time_lim=t_lim,
                device=self.device,
                batch_size=rad_batch
            )
        
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
        
    def train(self):
        max_steps = self.cfg.trainer.max_steps
        adam_epochs = self.cfg.optimizer.epochs
        
        # Initialize LBFGS if used
        if self.cfg.optimizer.lbfgs.use:
            log.info("Initializing LBFGS optimizer...")
            self.lbfgs_optimizer = optim.LBFGS(
                self.model.parameters(),
                lr=self.cfg.optimizer.lbfgs.lr,
                max_iter=self.cfg.optimizer.lbfgs.max_iter,
                history_size=self.cfg.optimizer.lbfgs.history_size
            )

        global_step = 0
        pbar = tqdm(range(max_steps), desc="Training Stages")
        
        for stage in pbar:
            self.run.log({"stage": stage}, step=global_step)
            t_stage_start = time.time()
            # 1. Sample (Once per stage)
            points = self.sampler.sample()
            
            # 2. RAD Sampling Logic (Once per stage, if enabled)
            if self.cfg.trainer.rad.use and stage > 0:
                rad_points_raw = self.rad_sampler.sample()
                # We need residuals on these candidate points
                # Note: We use the model from the end of previous stage
                res_dict = self.pde.residual(self.model, rad_points_raw)
                res_domain = res_dict['domain']
                
                n_rad = int(self.cfg.trainer.rad.ratio * self.cfg.trainer.batch_size.domain)
                rad_selected = self.sampler.rad_sampler(res_domain, rad_points_raw['domain'], n_rad)
                rad_selected = rad_selected.detach()
                
                current_domain = points['domain']
                n_keep = current_domain.shape[0] - n_rad
                if n_keep > 0:
                    keep_idx = torch.randperm(current_domain.shape[0])[:n_keep]
                    points['domain'] = torch.cat([current_domain[keep_idx], rad_selected], dim=0)
                else:
                    points['domain'] = rad_selected
                # save plot for rad_selected
                self.plotter.plot_scatter(
                    points=current_domain.cpu().numpy(),
                    rad_points=rad_selected.cpu().numpy(),
                    t_lim=self.pde.t_lim,
                    x_lim=self.pde.x_lim,
                    name=f"rad_selected_stage_{stage}",
                    epoch=global_step
                )

            # --- Adam Phase ---
            t_adam_start = time.time()
            for epoch in range(adam_epochs):
                # 3. Optimization Step
                self.optimizer.zero_grad()
                residuals = self.pde.residual(self.model, points)
                loss = 0
                log_dict = {}
                weights = self.cfg.pde.weights
                
                if 'domain' in residuals:
                    l = torch.mean(torch.square(residuals['domain']))
                    loss += weights.domain * l
                    log_dict['training/loss_domain'] = l.item()
                if 'boundary' in residuals:
                    l = torch.mean(torch.square(residuals['boundary']))
                    loss += weights.boundary * l
                    log_dict['training/loss_boundary'] = l.item()
                if 'initial' in residuals:
                    l = torch.mean(torch.square(residuals['initial']))
                    loss += weights.initial * l
                    log_dict['training/loss_initial'] = l.item()
                if 'initial_dt' in residuals:
                    l = torch.mean(torch.square(residuals['initial_dt']))
                    loss += weights.initial_dt * l
                    log_dict['training/loss_initial_dt'] = l.item()
                
                log_dict['training/loss_total'] = loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                global_step += 1
                
                # Logging
                log_dict['training/stage'] = stage
                self.run.log(log_dict, step=global_step)
                if global_step % 100 == 0:
                    pbar.set_postfix({'stage': stage, 'epoch': epoch, 'loss': f"{loss.item():.2e}"})
            
            self.run.log({"perf/time_adam": time.time() - t_adam_start}, step=global_step)

            # --- LBFGS Phase ---
            if self.cfg.optimizer.lbfgs.use:
                t_lbfgs_start = time.time()
                lbfgs_epochs = self.cfg.optimizer.lbfgs.epochs
                
                def closure():
                    nonlocal global_step
                    self.lbfgs_optimizer.zero_grad()
                    residuals = self.pde.residual(self.model, points)
                    loss = 0
                    log_dict = {}
                    weights = self.cfg.pde.weights
                    
                    if 'domain' in residuals:
                        l = torch.mean(torch.square(residuals['domain']))
                        loss += weights.domain * l
                        log_dict['training/loss_domain'] = l.item()
                    if 'boundary' in residuals:
                        l = torch.mean(torch.square(residuals['boundary']))
                        loss += weights.boundary * l
                        log_dict['training/loss_boundary'] = l.item()
                    if 'initial' in residuals:
                        l = torch.mean(torch.square(residuals['initial']))
                        loss += weights.initial * l
                        log_dict['training/loss_initial'] = l.item()
                    if 'initial_dt' in residuals:
                        l = torch.mean(torch.square(residuals['initial_dt']))
                        loss += weights.initial_dt * l
                        log_dict['training/loss_initial_dt'] = l.item()
                    
                    log_dict['training/loss_total'] = loss.item()

                    global_step += 1
                    log_dict['training/stage'] = stage
                    self.run.log(log_dict, step=global_step)
                    if global_step % 100 == 0:
                        pbar.set_postfix({'stage': stage, 'loss': f"{loss.item():.2e}"})

                    loss.backward()
                    return loss

                for epoch in range(lbfgs_epochs):
                    self.lbfgs_optimizer.step(closure)
                    
                self.run.log({"perf/time_lbfgs": time.time() - t_lbfgs_start}, step=global_step)

            
            
            self.run.log({"perf/time_stage": time.time() - t_stage_start}, step=global_step)
            # Evaluate & Checkpoint at the end of each Stage
            self.evaluate(stage=stage, step=global_step)
            self.save_checkpoint(stage)
            self.run.log({"perf/time_total": time.time() - t_stage_start}, step=global_step)
            

    def evaluate(self, stage, step):
        with torch.no_grad():
            # Create mesh
            # Assume 1D spatial for plotting
            if self.pde.__class__.__name__ == "BurgersPDE": 
                t_eval = self.pde.data['t']
                x_eval = self.pde.data['x']
            else:
                t_eval = np.linspace(self.cfg.pde.t_lim[0], self.cfg.pde.t_lim[1], 100)
                x_eval = np.linspace(self.cfg.pde.x_lim[0], self.cfg.pde.x_lim[1], 100)
            
            T, X = np.meshgrid(t_eval, x_eval)    
            points_np = np.stack([T.flatten(), X.flatten()], axis=1)
            points_tensor = torch.from_numpy(points_np).to(dtype=torch.get_default_dtype(), device=self.device)
            
            u_pred = self.model(points_tensor).cpu().numpy().reshape(T.shape)
            u_exact = self.pde.exact(points_tensor).cpu().numpy().reshape(T.shape)
            
            self.plotter.plot_solution(
                T,
                X,
                u_pred,
                title = None,
                name=f"prediction_step_{stage}"
            )
            if stage == 0:
                self.plotter.plot_solution(
                    T,
                    X,
                    u_exact,
                    title=None,
                    name=f"exact_solution"
                )
            # Log error metrics
            l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
            self.plotter.plot_solution(
                T,
                X,
                np.abs(u_pred - u_exact),
                title=f"{stage}-th error is {round(l2_error, 4)}",
                name=f"error_step_{stage}"
            )
            log_dict = {"val/L2error": l2_error}
            self.run.log(log_dict, step=step)
            log.info(f"Step {stage}: L2 Error = {l2_error:.2e}")

    def save_checkpoint(self, step):
        base_dir = os.path.join(os.getcwd(),'results','checkpoints')
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"checkpoint_{step}.pt")  
        torch.save(self.model.state_dict(), path)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float64)
    log.info(OmegaConf.to_yaml(cfg))
    run:wandb.run = setup_wandb(cfg)
    trainer = Trainer(cfg, run)
    trainer.train()

if __name__ == "__main__":
    main()
