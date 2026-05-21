import numpy as np

try:
    from .frac_weno_solver import frac_burgers_equation_solver
    from .initial_conditions import sine_ic, gaussian_ic, step_ic, multi_peak_ic
    from .boundary_conditions import dirichlet_bc, periodic_bc
    from .utils import (
        create_domain,
        plot_solution,
        animate_solution,
        analyze_results,
        compare_solvers,
        viscosity_study,
        create_output_directory,
        save_simulation_metadata,
        compute_error_norms,
        plot_error_distribution,
        compute_conservation_error,
        analyze_shock_properties,
        plot_shock_evolution,
        compute_spectral_properties,
        plot_spectrum,
    )
except ImportError:
    from frac_weno_solver import frac_burgers_equation_solver
    from initial_conditions import sine_ic, gaussian_ic, step_ic, multi_peak_ic
    from boundary_conditions import dirichlet_bc, periodic_bc
    from utils import (
        create_domain,
        plot_solution,
        animate_solution,
        analyze_results,
        compare_solvers,
        viscosity_study,
        create_output_directory,
        save_simulation_metadata,
        compute_error_norms,
        plot_error_distribution,
        compute_conservation_error,
        analyze_shock_properties,
        plot_shock_evolution,
        compute_spectral_properties,
        plot_spectrum,
    )

def run_simulation(L, Nx, Nt, T, nu, ic_func, bc_func, ic_name, bc_name, output_dir, al):
    """
    Run a simulation of the Burgers' equation with specified parameters.
    """
    print(f"Starting simulation for {ic_name} IC and {bc_name} BC...")
    try:
        x = create_domain(L, Nx)
        u0, du0= ic_func(x)
        
        t, u, _ = frac_burgers_equation_solver(u0, x, T, nu, bc_func, Nt, al, du0)
        
        print("Solver completed. Starting post-processing...")
        
        # 将模拟结果保存为字典格式
        print("Saving simulation data...")
        simulation_data = {
            't': t,
            'x': x,
            'u': u
        }
        data_filename = f"{output_dir}/data_{ic_name}_{bc_name}.npz"
        np.savez(data_filename, **simulation_data)
        print(f"Simulation data saved to {data_filename}")
        
        plot_title = f"Burgers' Equation: nu={nu}, IC={ic_name}, BC={bc_name}"
        print("Plotting solution...")
        plot_solution(x, t, u, title=plot_title, save_path=f"{output_dir}/solution_{ic_name}_{bc_name}.png")
        
        print("Creating animation...")
        try:
            animate_solution(x, t, u, title=plot_title, save_path=f"{output_dir}/animation_{ic_name}_{bc_name}.gif")
        except Exception as e:
            print(f"Warning: Failed to create animation. Error: {str(e)}")
        
        print(f"Simulation for {ic_name} IC and {bc_name} BC completed successfully.")
    except Exception as e:
        print(f"Error occurred during simulation or analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Set up parameters
    L = 1.0
    Nx = 1000
    Nt = 2000
    T = 1.20
    al= 1.75
    nu = 0.01/np.pi

    # Create output directory
    output_dir = r'./results/dw_burgers_175v3'
    
    # Save simulation metadata
    save_simulation_metadata(output_dir, L=L, Nx=Nx, Nt=Nt, T=T, nu=nu, al=al)

    # Define initial conditions
    ic_functions = {
        "Sine": sine_ic,
    }

    # Define boundary condition (only periodic BC is used in the tests)
    # bc_function = periodic_bc
    bc_function = dirichlet_bc
    # Run simulations for different initial conditions
    for ic_name, ic_func in ic_functions.items():
        print(f"\nRunning simulation with {ic_name} IC and Dirichlet BC...")
        run_simulation(L, Nx, Nt, T, nu, ic_func, bc_function, ic_name, "dirichlet", output_dir, al)

if __name__ == "__main__":
    main()
