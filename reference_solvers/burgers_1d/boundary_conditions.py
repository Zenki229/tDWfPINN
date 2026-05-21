import numpy as np
from typing import Callable

def dirichlet_bc(u: np.ndarray, left=0.0, right=0.0) -> np.ndarray:
    """
    Apply Dirichlet boundary conditions.
    
    Args:
    u (np.ndarray): Solution array
    left (float): Value at the left boundary
    right (float): Value at the right boundary
    
    Returns:
    np.ndarray: Updated solution array with applied boundary conditions
    """
    u[0] = left
    u[-1] = right
    return u

def periodic_bc(u: np.ndarray) -> np.ndarray:
    """
    Apply periodic boundary conditions.
    
    Args:
    u (np.ndarray): Solution array
    
    Returns:
    np.ndarray: Updated solution array with applied boundary conditions
    """
    u[0] = u[-2]
    u[-1] = u[1]
    return u

def neumann_bc(u: np.ndarray, left_derivative: float, right_derivative: float, dx: float) -> np.ndarray:
    """
    Apply Neumann boundary conditions.
    
    Args:
    u (np.ndarray): Solution array
    left_derivative (float): Derivative value at the left boundary
    right_derivative (float): Derivative value at the right boundary
    dx (float): Spatial step size
    
    Returns:
    np.ndarray: Updated solution array with applied boundary conditions
    """
    u[0] = u[1] - left_derivative * dx
    u[-1] = u[-2] + right_derivative * dx
    return u

def get_bc_function(bc_type: str, **kwargs) -> Callable:
    """
    Get the boundary condition function based on the specified type.
    
    Args:
    bc_type (str): Type of boundary condition ('dirichlet', 'periodic', or 'neumann')
    **kwargs: Additional arguments for the boundary condition function
    
    Returns:
    Callable: Boundary condition function
    """
    if bc_type == 'dirichlet':
        return lambda u: dirichlet_bc(u, kwargs.get('left', 0), kwargs.get('right', 0))
    elif bc_type == 'periodic':
        return periodic_bc
    elif bc_type == 'neumann':
        return lambda u: neumann_bc(u, kwargs.get('left_derivative', 0), kwargs.get('right_derivative', 0), kwargs.get('dx', 1))
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")