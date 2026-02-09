import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Dict, Optional
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class BasePlotter:
    def __init__(self, save_dir: str, cfg: Optional[DictConfig] = None):
        self.cfg = cfg
        self.save_dir = os.path.join(save_dir, "results")
        self.raw_dir = os.path.join(self.save_dir, "raw_data")
        self.img_dir = os.path.join(self.save_dir, "plots")
        try:
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.img_dir, exist_ok=True)
            log.info(f"Results directory created at: {self.save_dir}")
        except Exception as e:
            log.error(f"Failed to create results directory at {self.save_dir}: {e}")
            raise e

    def save_state(self, name: str, data: Dict[str, np.ndarray]):
        path = os.path.join(self.raw_dir, f"{name}.npz")
        try:
            np.savez_compressed(path, **data)
            log.info(f"Saved raw data to {path}")
        except Exception as e:
            log.error(f"Failed to save raw data to {path}: {e}")

    def _resolve_font_size(self, font_size: Optional[int]) -> int:
        if font_size is not None:
            return int(font_size)
        if self.cfg is not None and hasattr(self.cfg, "plot") and hasattr(self.cfg.plot, "font_size"):
            return int(self.cfg.plot.font_size)
        return 14

    def _jpg_enabled(self) -> bool:
        if self.cfg is not None and hasattr(self.cfg, "plot") and hasattr(self.cfg.plot, "jpg"):
            return bool(self.cfg.plot.jpg)
        return True

    def _resolve_dpi(self) -> int:
        if self.cfg is not None and hasattr(self.cfg, "plot") and hasattr(self.cfg.plot, "dpi"):
            return int(self.cfg.plot.dpi)
        return 100


class PlotlyPlotter(BasePlotter):
    def _save_figure(self, fig: go.Figure, filename_base: str):
        html_path = os.path.join(self.img_dir, f"{filename_base}.html")
        try:
            fig.write_html(html_path)
            log.info(f"Saved HTML plot to {html_path}")
        except Exception as e:
            log.error(f"Failed to save HTML plot to {html_path}: {e}")
        if self._jpg_enabled():
            jpg_path = os.path.join(self.img_dir, f"{filename_base}.jpg")
            fig.write_image(jpg_path)

    def plot_solution(
        self,
        t: np.ndarray,
        x: np.ndarray,
        values: np.ndarray,
        title: str,
        name: Optional[str] = None,
        font_size: Optional[int] = None
    ):
        if name is None:
            name = title.strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
            if not name:
                name = "plot"

        self.save_state(name, {"t": t, "x": x, "values": values})

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=values,
                x=t[0],
                y=x[:, 0],
                colorscale="Jet",
                zsmooth="best",
                colorbar=dict(tickformat=".1e", thickness=18)
            )
        )
        font_size = self._resolve_font_size(font_size)

        fig.update_layout(
            title_text=title,
            width=640,
            template="simple_white",
            xaxis_title="t",
            yaxis_title="x",
            margin=dict(l=60, r=40, t=60, b=60),
            font=dict(size=font_size),
            title_font=dict(size=font_size + 2)
        )
        fig.update_yaxes(scaleanchor="x")

        self._save_figure(fig, name)

    def plot_scatter(self, points: np.ndarray, name: str, epoch: int):
        filename_base = f"{name}_epoch_{epoch}"
        self.save_state(filename_base, {"points": points})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(size=2, color='red')))
        fig.update_layout(title=f"{name} Points at Epoch {epoch}", xaxis_title="t", yaxis_title="x")
        self._save_figure(fig, filename_base)


class PltPlotter(BasePlotter):
    def plot_solution(
        self,
        t: np.ndarray,
        x: np.ndarray,
        values: np.ndarray,
        title: str,
        name: Optional[str] = None,
        font_size: Optional[int] = None
    ):
        if name is None:
            name = title.strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
            if not name:
                name = "plot"

        self.save_state(name, {"t": t, "x": x, "values": values})

        font_size = self._resolve_font_size(font_size)
        plt.rcParams.update({'font.size': font_size})
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = ax.pcolormesh(t, x, values, shading='gouraud', cmap='jet')
        fig.colorbar(plot, ax=ax, format="%1.1e")
        if title: # if title is null, do not plot title
            ax.set_title(title)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        if self._jpg_enabled():
            jpg_path = os.path.join(self.img_dir, f"{name}.jpg")
            fig.savefig(jpg_path, dpi=self._resolve_dpi())
        plt.close(fig)

    def plot_scatter(self, points: np.ndarray, rad_points: np.ndarray, t_lim: list, x_lim: list, name: str, epoch: int):
        filename_base = f"{name}_epoch_{epoch}"
        self.save_state(filename_base, {"points": points, "rad_points": rad_points})
        font_size = self._resolve_font_size(None)
        plt.rcParams.update({'font.size': font_size})
        
        ts, te = t_lim[0], t_lim[1]
        xs, xe = x_lim[0], x_lim[1]
        
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        ax.set_xlim(ts - (te - ts) * 0.05, te + (te - ts) * 0.20)
        ax.set_ylim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.20)
        
        ax.scatter(points[:, 0], points[:, 1], c='b', marker='.', s=np.ones_like(points[:, 0]), alpha=0.3, label='uni')
        ax.scatter(rad_points[:, 0], rad_points[:, 1], c='r', marker='.', s=np.ones_like(rad_points[:, 0]), alpha=1.0, label='RAD')
        
        ax.legend(loc='upper right')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        if self._jpg_enabled():
            jpg_path = os.path.join(self.img_dir, f"{filename_base}.jpg")
            fig.savefig(jpg_path, dpi=self._resolve_dpi())
        plt.close(fig)
