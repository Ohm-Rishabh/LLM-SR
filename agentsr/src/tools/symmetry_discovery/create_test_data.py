#!/usr/bin/env python3
"""
Create a simple test HDF5 dataset for symmetry discovery.

This creates a harmonic oscillator dataset:
    dx1/dt = x2
    dx2/dt = -x1

This system has a rotational symmetry with Lie generator:
    G = [[0, -1], [1, 0]]
"""

import h5py
import numpy as np
import json

# Simulation parameters
n_trajs = 10
n_timesteps = 100
dt = 0.05

# Generate trajectories
trajectories_x1 = []
trajectories_x2 = []
trajectories_x1_t = []
trajectories_x2_t = []

for i in range(n_trajs):
    # Random initial conditions
    x1_0 = np.random.randn()
    x2_0 = np.random.randn()

    x1 = np.zeros(n_timesteps)
    x2 = np.zeros(n_timesteps)
    x1_t = np.zeros(n_timesteps)
    x2_t = np.zeros(n_timesteps)

    x1[0] = x1_0
    x2[0] = x2_0

    # Integrate using simple Euler method
    for t in range(n_timesteps - 1):
        x1_t[t] = x2[t]
        x2_t[t] = -x1[t]

        x1[t + 1] = x1[t] + dt * x1_t[t]
        x2[t + 1] = x2[t] + dt * x2_t[t]

    # Compute derivative at last timestep
    x1_t[-1] = x2[-1]
    x2_t[-1] = -x1[-1]

    trajectories_x1.append(x1)
    trajectories_x2.append(x2)
    trajectories_x1_t.append(x1_t)
    trajectories_x2_t.append(x2_t)

# Stack into arrays
x1_data = np.array(trajectories_x1)  # shape: (n_trajs, n_timesteps)
x2_data = np.array(trajectories_x2)
x1_t_data = np.array(trajectories_x1_t)
x2_t_data = np.array(trajectories_x2_t)

print(f"Generated data shapes:")
print(f"  x1: {x1_data.shape}")
print(f"  x2: {x2_data.shape}")
print(f"  x1_t: {x1_t_data.shape}")
print(f"  x2_t: {x2_t_data.shape}")

# Save to HDF5
output_file = "test_harmonic_oscillator.h5"

with h5py.File(output_file, 'w') as f:
    # Store variables
    f.create_dataset('x1', data=x1_data)
    f.create_dataset('x2', data=x2_data)
    f.create_dataset('x1_t', data=x1_t_data)
    f.create_dataset('x2_t', data=x2_t_data)

    # Store metadata as attributes
    f.attrs['independent_variables'] = json.dumps(['t'])
    f.attrs['dependent_variables'] = json.dumps(['x1', 'x2'])
    f.attrs['feature_variables'] = json.dumps(['x1', 'x2'])
    f.attrs['target_variables'] = json.dumps(['x1_t', 'x2_t'])

    # Optional: store ground truth symmetry for verification
    ground_truth_generator = [[0, -1], [1, 0]]
    f.attrs['ground_truth_lie_generator'] = json.dumps(ground_truth_generator)

print(f"\nSaved test dataset to: {output_file}")
print(f"Ground truth Lie generator (rotational symmetry):")
print(f"  [[0, -1],")
print(f"   [1,  0]]")
print(f"\nThis represents the infinitesimal generator of rotations in the (x1, x2) plane.")
