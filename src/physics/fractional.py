import numpy as np
from scipy.special import gamma, beta

try:
    from pymittagleffler import mittag_leffler
except ImportError:
    def mittag_leffler(z, alpha, beta):
        raise ImportError("pymittagleffler is required. Please install it.")

try:
    from pymittagleffler import roots_jacobi
except ImportError:
    # Fallback for roots_jacobi using scipy
    from scipy.special import roots_jacobi as scipy_roots_jacobi
    
    def roots_jacobi(n, alpha, beta):
        return scipy_roots_jacobi(n, alpha, beta)

def mitlef(alpha: float, beta: float, z: np.ndarray) -> np.ndarray:
    """
    Compute the Mittag-Leffler function.
    
    Args:
        alpha (float): Parameter alpha.
        beta (float): Parameter beta.
        z (np.ndarray): Input array.
        
    Returns:
        np.ndarray: Real part of Mittag-Leffler function values.
    """
    return np.real(mittag_leffler(z, alpha, beta))
