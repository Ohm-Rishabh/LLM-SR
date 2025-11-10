"""
Python implementation of simulate_ode.m
Simulates 7 ODE systems from Strogatz's book with the same parameters and initial conditions.
"""

import numpy as np
from scipy.integrate import solve_ivp
import os
import argparse


# Simulation parameters
TIME_SPAN = 10
SAMP_TIME = 0.1
N = 100
NUM_TRAJECTORIES = 4

# Set random seed for reproducibility (optional - remove if you want different random ICs each time)
# np.random.seed(42)


def bacterial_respiration(t, u):
    """Bacterial respiration (pg 288, Strogatz)"""
    x, y = u
    dxdt = 20 - x - (x * y) / (1 + 0.5 * x**2)
    dydt = 10 - (x * y) / (1 + 0.5 * x**2)
    return [dxdt, dydt]


def bar_magnets(t, u):
    """Bar magnets (p 286, Strogatz)"""
    x, y = u
    dxdt = 0.5 * np.sin(x - y) - np.sin(x)
    dydt = 0.5 * np.sin(y - x) - np.sin(y)
    return [dxdt, dydt]


def glider(t, u):
    """Glider (pg 188, Strogatz)"""
    x, y = u
    dxdt = -0.05 * x**2 - np.sin(y)
    dydt = x - np.cos(y) / x
    return [dxdt, dydt]


def lotka_volterra(t, u):
    """Lotka-Volterra"""
    x, y = u
    dxdt = 3 * x - 2 * x * y - x**2
    dydt = 2 * y - x * y - y**2
    return [dxdt, dydt]


def predator_prey(t, u):
    """Predator-prey (pg 288, Strogatz)"""
    x, y = u
    dxdt = x * (4 - x - y / (1 + x))
    dydt = y * (x / (1 + x) - 0.075 * y)
    return [dxdt, dydt]


def shear_flow(t, u):
    """Shear flow (p 192, Strogatz)"""
    x, y = u
    dxdt = np.cos(x) / np.tan(y)
    dydt = (np.cos(y)**2 + 0.1 * np.sin(y)**2) * np.sin(x)
    return [dxdt, dydt]


def van_der_pol(t, u):
    """Van der Pol (p 212, Strogatz)"""
    x, y = u
    dxdt = 10 * (y - (1/3) * (x**3 - x))
    dydt = -1/10 * x
    return [dxdt, dydt]


def simulate_system(ode_func, initial_conditions_list, system_name, noise_level=0.0):
    """
    Simulate an ODE system with multiple initial conditions.

    Parameters:
    - ode_func: ODE function to integrate
    - initial_conditions_list: List of initial conditions
    - system_name: Name of the system for output files
    - noise_level: Relative noise level (noise_std = noise_level * feature_std)

    Returns:
    - dataset1: [dx/dt, x, y] for each time point
    - dataset2: [dy/dt, x, y] for each time point
    """
    dataset1 = []
    dataset2 = []

    for ic in initial_conditions_list:
        # Time points
        t_eval = np.arange(0, TIME_SPAN, SAMP_TIME)

        # Solve ODE
        try:
            sol = solve_ivp(
                ode_func,
                [0, TIME_SPAN],
                ic,
                t_eval=t_eval,
                method='RK45',
                dense_output=True
            )

            # Extract first N points
            x_clean = sol.y[0, :N]
            y_clean = sol.y[1, :N]
            t = sol.t[:N]

            # Add relative white noise to trajectories
            x_std = np.std(x_clean)
            y_std = np.std(y_clean)
            x_noisy = x_clean + np.random.normal(0, noise_level * x_std, N)
            y_noisy = y_clean + np.random.normal(0, noise_level * y_std, N)

            # Compute derivatives using finite differences (numpy.gradient) on noisy data
            dxdt = np.gradient(x_noisy, t)
            dydt = np.gradient(y_noisy, t)

            # Append to datasets
            for i in range(N):
                dataset1.append([dxdt[i], x_noisy[i], y_noisy[i]])
                dataset2.append([dydt[i], x_noisy[i], y_noisy[i]])

        except Exception as e:
            print(f"Warning: Failed to simulate {system_name} with IC {ic}: {e}")
            continue

    return np.array(dataset1), np.array(dataset2)


def main():
    """Main function to simulate all ODE systems and save datasets."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Simulate ODE systems from Strogatz with optional noise'
    )
    parser.add_argument('--noise', type=float, default=0.0, help='Relative noise level (noise_std = noise_level * feature_std). Default: 0.0')
    parser.add_argument('--output_dir', type=str, default="data/strogatz-ode")
    args = parser.parse_args()

    noise_level = args.noise

    # Create output directory if it doesn't exist
    output_dir = f"{args.output_dir}/noise{args.noise:.2f}"
    os.makedirs(output_dir, exist_ok=True)

    print("Simulating ODE systems from Strogatz...")
    print(f"Time span: {TIME_SPAN}, Sampling time: {SAMP_TIME}, N: {N}")
    print(f"Noise level: {noise_level} (relative)\n")

    # Define all ODE systems with their configurations
    systems = [
        {
            'name': 'Bacterial Respiration',
            'func': bacterial_respiration,
            'prefix': 'bacres',
            'ic_generator': lambda: [5 + np.random.normal(0, 1),
                                    10 + np.random.normal(0, np.sqrt(0.1))]
        },
        {
            'name': 'Bar Magnets',
            'func': bar_magnets,
            'prefix': 'barmag',
            'ic_generator': lambda: [2 * np.pi * np.random.rand(),
                                    2 * np.pi * np.random.rand()]
        },
        {
            'name': 'Glider',
            'func': glider,
            'prefix': 'glider',
            'ic_generator': lambda: [5 + np.random.normal(0, 1),
                                    0 + np.random.normal(0, np.sqrt(0.1))]
        },
        {
            'name': 'Lotka-Volterra',
            'func': lotka_volterra,
            'prefix': 'lotkavolterra',
            'ics': [[1, 3], [4, 1], [8, 2], [3, 3]]  # Fixed initial conditions
        },
        {
            'name': 'Predator-Prey',
            'func': predator_prey,
            'prefix': 'predprey',
            'ic_generator': lambda: [5 + np.random.normal(0, 1),
                                    10 + np.random.normal(0, np.sqrt(0.1))]
        },
        {
            'name': 'Shear Flow',
            'func': shear_flow,
            'prefix': 'shearflow',
            'ic_generator': lambda: [2 * np.pi * np.random.rand() - np.pi,
                                    np.pi * np.random.rand() - np.pi/2]
        },
        {
            'name': 'Van der Pol',
            'func': van_der_pol,
            'prefix': 'vdp',
            'ic_generator': lambda: [np.random.rand(), np.random.rand()]
        }
    ]

    # Simulate all systems
    for idx, system in enumerate(systems, 1):
        print(f"{idx}. {system['name']}...")

        # Generate or use fixed initial conditions
        if 'ics' in system:
            # Use fixed initial conditions
            ics = system['ics']
        else:
            # Generate random initial conditions
            ics = [system['ic_generator']() for _ in range(NUM_TRAJECTORIES)]

        # Simulate the system
        dataset1, dataset2 = simulate_system(
            system['func'],
            ics,
            system['name'].lower().replace(' ', '_').replace('-', '_'),
            noise_level
        )

        # Save datasets
        prefix = system['prefix']
        np.savetxt(f"{output_dir}/{prefix}1.csv", dataset1, delimiter=",",
                   header="dxdt,x,y", comments='')
        np.savetxt(f"{output_dir}/{prefix}2.csv", dataset2, delimiter=",",
                   header="dydt,x,y", comments='')

        print(f"   Saved: {prefix}1.csv ({dataset1.shape[0]} rows), "
              f"{prefix}2.csv ({dataset2.shape[0]} rows)")

    print("\nAll simulations completed successfully!")
    print(f"Output files saved to: {output_dir}/")


if __name__ == "__main__":
    main()
