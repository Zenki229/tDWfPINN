import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Dict
from matplotlib.animation import FuncAnimation
import os
from datetime import datetime
import time
import shutil
def create_domain(L: float, Nx: int) -> np.ndarray:
    """
    Create a spatial domain.
    
    Args:
    L (float): Domain length
    Nx (int): Number of spatial grid points
    
    Returns:
    np.ndarray: Spatial grid
    """
    return np.linspace(-L, L, Nx)

def plot_solution(x: np.ndarray, t: np.ndarray, u: np.ndarray, 
                  title: str = "Burgers' Equation Solution",
                  save_path: str = None):
    """Plot the solution of Burgers' equation."""
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, x, u.T, shading='gouraud', cmap='jet')
    plt.colorbar(label='u')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def animate_solution(x: np.ndarray, t: np.ndarray, u: np.ndarray, 
                     title: str = "Burgers' Equation Solution Animation",
                     save_path: str = None, max_frames: int = 200, timeout: int = 300):
    """
    Create an animation of the Burgers' equation solution.
    
    Args:
    x (np.ndarray): Spatial grid
    t (np.ndarray): Time grid
    u (np.ndarray): Solution array (2D: time x space)
    title (str): Animation title
    save_path (str): Path to save the animation
    max_frames (int): Maximum number of frames to use in the animation
    timeout (int): Maximum time (in seconds) to spend on creating the animation
    """
    print(f"Starting animation creation. Max frames: {max_frames}, Timeout: {timeout} seconds")
    start_time = time.time()

    # Reduce the number of frames if necessary
    num_frames = min(len(t), max_frames)
    frame_indices = np.linspace(0, len(t) - 1, num_frames, dtype=int)

    # Reduce spatial resolution if the array is too large
    max_spatial_points = 500
    if len(x) > max_spatial_points:
        skip = len(x) // max_spatial_points
        x = x[::skip]
        u = u[:, ::skip]

    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(u.min(), u.max())
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(title)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        if time.time() - start_time > timeout:
            raise TimeoutError("Animation creation timed out")
        idx = frame_indices[i]
        line.set_data(x, u[idx, :])
        if i % 10 == 0:
            print(f"Animating frame {i}/{num_frames}")
        return (line,)

    try:
        anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True)

        if save_path:
            print(f"Saving animation to {save_path}")
            anim.save(save_path, writer='pillow', fps=30)
            print("Animation saved successfully")
        plt.close(fig)
    except TimeoutError as e:
        print(f"Warning: {str(e)}. Skipping animation.")
    except Exception as e:
        print(f"Error during animation creation: {str(e)}")
    finally:
        plt.close(fig)

    print(f"Animation process completed in {time.time() - start_time:.2f} seconds")

def compare_solvers(L: float, Nx: int, Nt: int, T: float, nu: float, 
                    ic_func: Callable, bc_func: Callable, 
                    solvers: List[Tuple[str, Callable]], save_dir: str = None) -> None:
    """Compare different solver implementations."""
    x = create_domain(L, Nx)
    u0 = ic_func(x)
    
    fig, axes = plt.subplots(len(solvers), 1, figsize=(10, 5*len(solvers)), sharex=True)
    fig.suptitle(f"Comparison of Burgers Equation Solvers (nu={nu})")
    
    for i, (name, solver) in enumerate(solvers):
        t, u, _ = solver(u0, x, T, nu, bc_func, Nt)
        im = axes[i].pcolormesh(x, t, u, shading='auto', cmap='viridis')
        axes[i].set_ylabel('t')
        axes[i].set_title(f"{name} Solver")
        plt.colorbar(im, ax=axes[i])
    
    axes[-1].set_xlabel('x')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"solver_comparison_nu_{nu}.png")
        plt.savefig(save_path)
    plt.close()

def viscosity_study(L: float, Nx: int, Nt: int, T: float, 
                    ic_func: Callable, bc_func: Callable, 
                    solver: Callable, viscosities: List[float], save_dir: str = None) -> None:
    """Study the effect of viscosity on the solution stability."""
    x = create_domain(L, Nx)
    u0 = ic_func(x)
    
    fig, axes = plt.subplots(len(viscosities), 1, figsize=(10, 5*len(viscosities)), sharex=True)
    fig.suptitle("Effect of Viscosity on Burgers Equation Solution")
    
    for i, nu in enumerate(viscosities):
        try:
            t, u, _ = solver(u0, x, T, nu, bc_func, Nt)
            im = axes[i].pcolormesh(x, t, u, shading='auto', cmap='viridis')
            axes[i].set_ylabel('t')
            axes[i].set_title(f"nu = {nu}")
            plt.colorbar(im, ax=axes[i])
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Solver failed: {str(e)}", ha='center', va='center')
    
    axes[-1].set_xlabel('x')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "viscosity_study.png")
        plt.savefig(save_path)
    plt.close()

def analyze_results(x: np.ndarray, t: np.ndarray, u: np.ndarray, save_dir: str = None):
    """Analyze the results of the Burgers' equation solution."""
    # Compute and plot total variation
    tv = np.sum(np.abs(np.diff(u, axis=1)), axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(t, tv)
    plt.xlabel('Time')
    plt.ylabel('Total Variation')
    plt.title("Total Variation over Time")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "total_variation.png")
        plt.savefig(save_path)
    plt.close()
    
    # Compute and plot energy
    energy = np.sum(u**2, axis=1) * (x[1] - x[0])
    plt.figure(figsize=(10, 6))
    plt.plot(t, energy)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title("Energy over Time")
    
    if save_dir:
        save_path = os.path.join(save_dir, "energy.png")
        plt.savefig(save_path)
    plt.close()
    
    # Compute and print shock formation time (if applicable)
    du_dx = np.gradient(u, x[1] - x[0], axis=1)
    shock_threshold = 50  # Adjust this value as needed
    
    shock_formation_times = t[np.any(np.abs(du_dx) > shock_threshold, axis=1)]
    
    if len(shock_formation_times) > 0:
        print(f"Approximate shock formation time: {shock_formation_times[0]}")
    else:
        print("No shock formation detected")

def plot_convergence(dx_values: np.ndarray, errors: Dict[str, np.ndarray], save_path: str = None):
    """Plot convergence study results."""
    plt.figure(figsize=(10, 6))
    for error_type, error_values in errors.items():
        plt.loglog(dx_values, error_values, 'o-', label=f'{error_type} Error')
        rate = compute_convergence_rate(dx_values, error_values)
        plt.loglog(dx_values, error_values[0] * (dx_values / dx_values[0])**rate, '--', 
                   label=f'{error_type} Rate: {rate:.2f}')
    
    plt.xlabel('Grid Size (dx)')
    plt.ylabel('Error')
    plt.title('Convergence Study')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_error_distribution(x: np.ndarray, error: np.ndarray, title: str, save_path: str = None):
    """Plot the spatial distribution of error."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, error)
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_shock_evolution(x: np.ndarray, t: np.ndarray, u: np.ndarray, save_path: str = None):
    """Plot the evolution of shock formation and propagation."""
    plt.figure(figsize=(12, 8))
    plt.contourf(x, t, u, levels=20, cmap='viridis')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Shock Evolution in Burgers Equation')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_spectrum(freq: np.ndarray, spectrum: np.ndarray, save_path: str = None):
    """Plot the power spectrum of the solution."""
    plt.figure(figsize=(10, 6))
    plt.loglog(freq[1:len(freq)//2], spectrum[1:len(freq)//2])  # Exclude zero frequency
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Power Spectrum')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_output_directory():
    """
    Create a timestamped output directory for saving results.
    
    Returns:
    str: Path to the created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"burgers_equation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def compute_convergence_rate(dx_values: np.ndarray, errors: np.ndarray) -> float:
    """
    Compute the convergence rate based on grid sizes and corresponding errors.
    """
    log_dx = np.log(dx_values)
    log_errors = np.log(errors)
    slope, _ = np.polyfit(log_dx, log_errors, 1)
    return slope

def compute_error_norms(u_numerical: np.ndarray, u_exact: np.ndarray) -> Dict[str, float]:
    """
    Compute various error norms between numerical and exact solutions.
    """
    error = np.abs(u_numerical - u_exact)
    return {
        'L1': np.mean(error),
        'L2': np.sqrt(np.mean(error**2)),
        'Linf': np.max(error),
        'RMS': np.sqrt(np.mean(error**2))
    }

def compute_conservation_error(u: np.ndarray, dx: float) -> float:
    """
    Compute the conservation error (change in total mass) over time.
    """
    mass = np.sum(u, axis=1) * dx
    return np.max(np.abs(mass - mass[0]))

def analyze_shock_properties(x: np.ndarray, t: np.ndarray, u: np.ndarray, shock_threshold: float = 10):
    """
    Analyze properties of shocks in the solution.
    """
    du_dx = np.gradient(u, x[1] - x[0], axis=1)
    shock_locations = np.abs(du_dx) > shock_threshold
    
    shock_formation_time = t[np.any(shock_locations, axis=1)][0] if np.any(shock_locations) else None
    shock_strength = np.max(np.abs(du_dx))
    shock_width = np.min(np.diff(x)[shock_locations]) if np.any(shock_locations) else None
    
    return {
        'formation_time': shock_formation_time,
        'max_strength': shock_strength,
        'min_width': shock_width
    }

def compute_spectral_properties(x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral properties of the solution.
    """
    n = len(x)
    dx = x[1] - x[0]
    freq = np.fft.fftfreq(n, dx)
    spectrum = np.abs(np.fft.fft(u))**2
    return freq, spectrum

def save_simulation_metadata(output_dir: str, **kwargs):
    """
    Save simulation metadata to a text file.
    
    Args:
    output_dir (str): Directory to save the metadata file
    **kwargs: Simulation parameters to save
    """
    if os.path.exists(output_dir):
            print(f"Overwriting existing directory: {output_dir}")
            time.sleep(2)
            # 使用 shutil.rmtree 删除整个目录及其内容
            shutil.rmtree(output_dir)
            # 重新创建目录
            os.makedirs(output_dir, exist_ok=True)
    else:
        # 如果目录不存在，则创建它（包括任何必要的父目录）
        os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "simulation_metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
