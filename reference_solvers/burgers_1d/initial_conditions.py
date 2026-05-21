import numpy as np

def sine_ic(x: np.ndarray) -> np.ndarray:
    """
    Sinusoidal initial condition.
    
    Args:
    x (np.ndarray): Spatial grid
    
    Returns:
    np.ndarray: Initial condition values
    """
    # return -np.sin(np.pi*x), np.zeros_like(x)
    return -np.sin(np.pi*x), 0.0*np.sin(np.pi*x)

def step_ic(x: np.ndarray) -> np.ndarray:
    """
    Step function initial condition.
    
    Args:
    x (np.ndarray): Spatial grid
    
    Returns:
    np.ndarray: Initial condition values
    """
    return np.where(x < np.pi, 1.0, 0.0)

def gaussian_ic(x: np.ndarray, mu: float = np.pi, sigma: float = np.pi/4) -> np.ndarray:
    """
    Gaussian initial condition.
    
    Args:
    x (np.ndarray): Spatial grid
    mu (float): Mean of the Gaussian (default: pi)
    sigma (float): Standard deviation of the Gaussian (default: pi/4)
    
    Returns:
    np.ndarray: Initial condition values
    """
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

def multi_peak_ic(x: np.ndarray, num_peaks: int = 3) -> np.ndarray:
    """
    Multi-peak initial condition.
    
    Args:
    x (np.ndarray): Spatial grid
    num_peaks (int): Number of peaks (default: 3)
    
    Returns:
    np.ndarray: Initial condition values
    """
    return sum(np.sin(i * x) for i in range(1, num_peaks + 1)) / num_peaks

def random_ic(x: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Random initial condition.
    
    Args:
    x (np.ndarray): Spatial grid
    seed (int): Random seed for reproducibility (default: None)
    
    Returns:
    np.ndarray: Initial condition values
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(len(x))