import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
import numpy as np
import os
import logging
from tqdm import tqdm
import wandb

from src.utils.experiments import set_seed, setup_wandb
from src.models.net import MLP
from src.physics.dw_pde import DWForward
from src.data.sampler import TimeSpaceSampler
from src.vis.plotter import Plotter

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup
        set_seed(cfg.seed)
        setup_wandb(cfg)
        
        # Components
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        self.pde = DWForward(cfg, self.device)
        
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
        
        batch_size = OmegaConf.to_container(cfg.training.batch_size, resolve=True)
        self.sampler = TimeSpaceSampler(
            spatial_lim=x_lim,
            time_lim=t_lim,
            device=self.device,
            batch_size=batch_size
        )
        
        if cfg.training.rad.use:
            rad_batch = OmegaConf.to_container(cfg.training.rad.batch, resolve=True)
            self.rad_sampler = TimeSpaceSampler(
                spatial_lim=x_lim,
                time_lim=t_lim,
                device=self.device,
                batch_size=rad_batch
            )
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.optimizer.lr)
        
        # Plotter
        self.plotter = Plotter(os.getcwd())
        
    def train(self):
        max_steps = self.cfg.training.max_steps
        pbar = tqdm(range(max_steps), desc="Training")
        
        for step in pbar:
            # 1. Sample
            points = self.sampler.sample()
            
            # 2. RAD Sampling Logic
            if self.cfg.training.rad.use and step > 0: # Maybe not every step? Original: every step.
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
                
                n_rad = int(self.cfg.training.rad.ratio * self.cfg.training.batch_size.domain)
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
            
            self.plotter.plot_solution(T, X, u_pred, u_exact, step)
            
            # Log error metrics
            l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
            wandb.log({"L2_Relative_Error": l2_error}, step=step)
            log.info(f"Step {step}: L2 Error = {l2_error:.2e}")

    def save_checkpoint(self, step):
        path = os.path.join(os.getcwd(), f"checkpoint_{step}.pt")
        torch.save(self.model.state_dict(), path)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
