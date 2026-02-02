import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
import json
import logging

log = logging.getLogger(__name__)

class Plotter:
    def __init__(self, save_dir: str):
        """
        Initialize the Plotter.
        
        Args:
            save_dir (str): Base directory for saving results.
        """
        self.save_dir = os.path.join(save_dir, "results")
        self.raw_dir = os.path.join(self.save_dir, "raw_data")
        self.img_dir = os.path.join(self.save_dir, "plots")
        
        # Ensure directories exist
        try:
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.img_dir, exist_ok=True)
            log.info(f"Results directory created at: {self.save_dir}")
        except Exception as e:
            log.error(f"Failed to create results directory at {self.save_dir}: {e}")
            raise e

    def save_state(self, name: str, data: Dict[str, np.ndarray]):
        """
        Save raw plotting data to NPZ.
        """
        path = os.path.join(self.raw_dir, f"{name}.npz")
        try:
            np.savez_compressed(path, **data)
            log.info(f"Saved raw data to {path}")
        except Exception as e:
            log.error(f"Failed to save raw data to {path}: {e}")

    def _save_figure(self, fig: go.Figure, filename_base: str):
        """
        Helper to save figure as HTML and JPG.
        """
        # Save HTML
        html_path = os.path.join(self.img_dir, f"{filename_base}.html")
        try:
            fig.write_html(html_path)
            log.info(f"Saved HTML plot to {html_path}")
        except Exception as e:
            log.error(f"Failed to save HTML plot to {html_path}: {e}")

        # Save JPG
        jpg_path = os.path.join(self.img_dir, f"{filename_base}.jpg")
        try:
            # Use kaleido for static image export
            fig.write_image(jpg_path, format="jpg", engine="kaleido")
            log.info(f"Saved JPG plot to {jpg_path}")
        except Exception as e:
            log.error(f"Failed to save JPG plot to {jpg_path}: {e}")
            # Fallback warning if kaleido is missing/broken
            log.warning("Ensure 'kaleido' is installed for static image export.")

    def plot_solution(self, t: np.ndarray, x: np.ndarray, 
                      u_pred: np.ndarray, u_exact: np.ndarray, 
                      epoch: int):
        """
        Plot solution heatmap comparison.
        t, x are meshgrids.
        """
        err = np.abs(u_pred - u_exact)
        name = f"solution_epoch_{epoch}"
        
        # Save raw data first
        self.save_state(name, {
            "t": t, "x": x, "u_pred": u_pred, "u_exact": u_exact, "error": err
        })

        # Create Plotly figure
        # 1x3 subplots: Pred, Exact, Error
        fig = make_subplots(rows=1, cols=3, 
                            subplot_titles=("Prediction", "Exact", "Abs Error"),
                            shared_yaxes=True)

        # Prediction
        fig.add_trace(go.Heatmap(z=u_pred, x=t[0], y=x[:,0], colorscale='Viridis', colorbar=dict(len=0.3, y=0.8)), row=1, col=1)
        
        # Exact
        fig.add_trace(go.Heatmap(z=u_exact, x=t[0], y=x[:,0], colorscale='Viridis', colorbar=dict(len=0.3, y=0.5)), row=1, col=2)
        
        # Error
        fig.add_trace(go.Heatmap(z=err, x=t[0], y=x[:,0], colorscale='Plasma', colorbar=dict(len=0.3, y=0.2)), row=1, col=3)

        fig.update_layout(title_text=f"Epoch {epoch} Solution Analysis", height=500, width=1500)
        
        self._save_figure(fig, name)

    def plot_scatter(self, points: np.ndarray, name: str, epoch: int):
        """
        Scatter plot for RAD points.
        points: N*2 [t, x]
        """
        filename_base = f"{name}_epoch_{epoch}"
        self.save_state(filename_base, {"points": points})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(size=2, color='red')))
        fig.update_layout(title=f"{name} Points at Epoch {epoch}", xaxis_title="t", yaxis_title="x")
        
        self._save_figure(fig, filename_base)
