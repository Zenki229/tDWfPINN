import numpy as np
from typing import Callable, Tuple

def weno5(v: np.ndarray) -> float:
    """
    Compute the WENO5 reconstruction.
    """
    epsilon = 1e-6
    gamma1, gamma2, gamma3 = 1/10, 6/10, 3/10
    v1, v2, v3, v4, v5 = v[0], v[1], v[2], v[3], v[4]
    
    beta1 = 13/12 * (v1 - 2*v2 + v3)**2 + 1/4 * (v1 - 4*v2 + 3*v3)**2
    beta2 = 13/12 * (v2 - 2*v3 + v4)**2 + 1/4 * (v2 - v4)**2
    beta3 = 13/12 * (v3 - 2*v4 + v5)**2 + 1/4 * (3*v3 - 4*v4 + v5)**2
    
    beta1 += epsilon
    beta2 += epsilon
    beta3 += epsilon
    
    alpha1 = gamma1 / beta1**2
    alpha2 = gamma2 / beta2**2
    alpha3 = gamma3 / beta3**2
    alpha_sum = alpha1 + alpha2 + alpha3
    
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum
    w3 = alpha3 / alpha_sum
    
    q1 = 1/3 * v1 - 7/6 * v2 + 11/6 * v3
    q2 = -1/6 * v2 + 5/6 * v3 + 1/3 * v4
    q3 = 1/3 * v3 + 5/6 * v4 - 1/6 * v5
    
    return w1 * q1 + w2 * q2 + w3 * q3

def weno_scheme(u: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the WENO scheme to compute spatial derivatives.
    """
    N = len(u)
    u_padded = np.pad(u, 3, mode='wrap')
    du_dx = np.zeros(N)
    
    for i in range(N):
        v_minus = u_padded[i:i+5]
        v_plus = u_padded[i+1:i+6]
        u_minus = weno5(v_minus)
        u_plus = weno5(v_plus)
        du_dx[i] = (u_plus - u_minus) / dx
    
    d2u_dx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
    
    return du_dx, d2u_dx2

def lax_friedrichs_flux(u_minus: np.ndarray, u_plus: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the Lax-Friedrichs flux for shock capturing.

    Args:
    u_minus (np.ndarray): Left state
    u_plus (np.ndarray): Right state
    alpha (float): Maximum wave speed

    Returns:
    np.ndarray: Lax-Friedrichs flux
    """
    f_minus = 0.5 * u_minus**2
    f_plus = 0.5 * u_plus**2
    return 0.5 * (f_minus + f_plus - alpha * (u_plus - u_minus))

def ssprk3_time_integration(u: np.ndarray, dt: float, rhs_func: Callable) -> np.ndarray:
    """
    Perform 3rd order Strong Stability Preserving Runge-Kutta time integration.

    Args:
    u (np.ndarray): Current solution
    dt (float): Time step
    rhs_func (Callable): Function to compute the right-hand side

    Returns:
    np.ndarray: Updated solution
    """
    u1 = u + dt * rhs_func(u)
    u2 = 3/4 * u + 1/4 * (u1 + dt * rhs_func(u1))
    u3 = 1/3 * u + 2/3 * (u2 + dt * rhs_func(u2))
    return u3

def compute_adaptive_dt(u: np.ndarray, dx: float, nu: float, cfl: float) -> float:
    """
    Compute an adaptive time step based on CFL condition and viscosity.

    Args:
    u (np.ndarray): Current solution
    dx (float): Spatial step size
    nu (float): Viscosity coefficient
    cfl (float): CFL number

    Returns:
    float: Adaptive time step
    """
    max_velocity = np.max(np.abs(u))
    dt_adv = cfl * dx / (max_velocity + 1e-8)
    dt_diff = cfl * dx**2 / (nu + 1e-8)
    return min(dt_adv, dt_diff)

def burgers_equation_solver(u0: np.ndarray, x: np.ndarray, T: float, nu: float, 
                            bc_func: Callable, Nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Burgers' equation using WENO scheme, Lax-Friedrichs flux splitting, and SSPRK3 time integration.

    Args:
    u0 (np.ndarray): Initial condition
    x (np.ndarray): Spatial grid
    T (float): Final time
    nu (float): Viscosity coefficient
    bc_func (Callable): Boundary condition function
    Nt (int): Maximum number of time steps

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Time grid, solution array, and error array
    """
    Nx = len(x)
    dx = x[1] - x[0]
    
    u = np.zeros((Nt+1, Nx))
    u[0] = u0
    
    t = np.zeros(Nt+1)
    errors = np.zeros(Nt+1)
    
    print(f"Starting solver with Nx={Nx}, Nt={Nt}, dx={dx:.6f}")
    
    def rhs(u):
        u_padded = np.pad(u, 1, mode='wrap')
        u_minus = u_padded[:-2]
        u = u_padded[1:-1]  # This is the centered u
        u_plus = u_padded[2:]
        alpha = np.max(np.abs(u))
        f_minus = 0.5 * u_minus**2
        f_plus = 0.5 * u_plus**2
        f = 0.5 * u**2
        flux_left = 0.5 * (f + f_minus - alpha * (u - u_minus))
        flux_right = 0.5 * (f_plus + f - alpha * (u_plus - u))
        du_dx = (flux_right - flux_left) / dx
        d2u_dx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
        return -du_dx + nu * d2u_dx2

    try:
        n = 0
        while t[n] < T and n < Nt:
            dt = compute_adaptive_dt(u[n], dx, nu, cfl=0.5)
            dt = min(dt, T - t[n])
            
            u[n+1] = ssprk3_time_integration(u[n], dt, rhs)
            u[n+1] = bc_func(u[n+1])
            
            t[n+1] = t[n] + dt
            n += 1
            
            if n % (Nt // 10) == 0:
                print(f"Step {n}, t={t[n]:.6f}, dt={dt:.6f}, Max u: {np.max(u[n]):.6f}, Min u: {np.min(u[n]):.6f}")
    
    except Exception as e:
        print(f"Error occurred during solving: {str(e)}")
        raise
    
    print("Solver completed successfully")
    return t[:n+1], u[:n+1], errors[:n+1]

def compute_mass(u: np.ndarray, dx: float) -> float:
    """
    Compute the total mass of the solution.
    """
    return np.sum(u) * dx

# Keep the basic_weno_solver function for comparison
def basic_weno_solver(u0: np.ndarray, x: np.ndarray, T: float, nu: float, 
                      bc_func: Callable, Nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Basic WENO solver with improved stability.
    """
    Nx = len(x)
    dx = x[1] - x[0]
    u = np.zeros((Nt+1, Nx))
    u[0] = u0
    t = np.zeros(Nt+1)
    
    n = 0
    while t[n] < T and n < Nt:
        dt = compute_adaptive_dt(u[n], dx, nu, cfl=0.5)
        dt = min(dt, T - t[n])
        
        du_dx, d2u_dx2 = weno_scheme(u[n], dx)
        u[n+1] = u[n] - dt * u[n] * du_dx + nu * dt * d2u_dx2
        u[n+1] = bc_func(u[n+1])
        
        t[n+1] = t[n] + dt
        n += 1
        
        if n % (Nt // 10) == 0:
            print(f"Basic WENO Step {n}, t={t[n]:.6f}, dt={dt:.6f}, Max u: {np.max(u[n]):.6f}, Min u: {np.min(u[n]):.6f}")
    
    return t[:n+1], u[:n+1], None