# Symmetry Discovery Tool

This tool discovers linear continuous symmetries in ODE/PDE systems using Lie group theory and neural network surrogates.

## Overview

The symmetry discovery tool:
1. Loads ODE/PDE data from HDF5 files
2. Trains a surrogate neural network (3-layer MLP) to approximate the system dynamics
3. Discovers Lie symmetries by optimizing a Lie generator matrix
4. Returns the discovered symmetry and optimization loss

## Theory

### Lie Symmetries

A **Lie symmetry** of a differential equation is a transformation that maps solutions to solutions. For continuous symmetries, these transformations form a Lie group, characterized by their infinitesimal generators (vector fields).

For a system of ODEs:
```
dx/dt = f(t, x)
```

A vector field `v` is a symmetry if:
```
Jacobian(f) · v(x) = 0
```

This tool optimizes for `v` by minimizing `||Jacobian(f) · v(x)||²`.

### Lie Generator

The Lie generator is parameterized as a `q × q` matrix `G`, where `q` is the number of dependent variables. This matrix generates infinitesimal transformations:
```
v(x) = G · x
```

The discovered matrix can be exponentiated to obtain finite transformations:
```
x' = exp(ε·G) · x  for small ε
```

### Prolongation

The **prolongation** of a vector field extends the symmetry from the base variables to their derivatives. The tool provides an interface `compute_prolongation()` that should be implemented with the specific prolongation formula for your system.

For first-order ODEs, a common prolongation formula is:
```
v^(1) = D_t(v) + (∂f/∂x)·v
```

where `D_t` is the total derivative with respect to time.

## File Structure

```
symmetry_discovery/
├── README.md              # This file
├── run.sh                 # Bash entry point
├── tool.py                # Main implementation
├── create_test_data.py    # Test data generator
├── test_tool.sh           # Test script
└── test_harmonic_oscillator.h5  # Example dataset
```

## Data Format

### HDF5 File Structure

The input HDF5 file must contain:

**Datasets** (one per variable):
- Each variable as a separate dataset (e.g., "x1", "x2", "x1_t", "x2_t")
- Shape: `(n_trajectories, n_timesteps)` or `(n_trajectories, n_timesteps, dim)`

**Attributes** (metadata):
- `independent_variables`: JSON list of independent variable names (e.g., ["t"])
- `dependent_variables`: JSON list of dependent variable names (e.g., ["x1", "x2"])
- `feature_variables`: JSON list of feature variable names (e.g., ["x1", "x2"])
- `target_variables`: JSON list of target variable names (e.g., ["x1_t", "x2_t"])

### Example: Creating a Dataset

```python
import h5py
import numpy as np
import json

# Generate or load your data
x1_data = ...  # shape: (n_trajs, n_timesteps)
x2_data = ...
x1_t_data = ...  # time derivatives
x2_t_data = ...

# Save to HDF5
with h5py.File('my_data.h5', 'w') as f:
    # Store variables
    f.create_dataset('x1', data=x1_data)
    f.create_dataset('x2', data=x2_data)
    f.create_dataset('x1_t', data=x1_t_data)
    f.create_dataset('x2_t', data=x2_t_data)

    # Store metadata
    f.attrs['independent_variables'] = json.dumps(['t'])
    f.attrs['dependent_variables'] = json.dumps(['x1', 'x2'])
    f.attrs['feature_variables'] = json.dumps(['x1', 'x2'])
    f.attrs['target_variables'] = json.dumps(['x1_t', 'x2_t'])
```

## Usage

### Within Agent-SR Framework

The tool is called via the LLM's tool selection mechanism. Example tool call:

```json
{
  "tool_name": "symmetry_discovery",
  "args": {
    "data_file": "harmonic_oscillator.h5",
    "surrogate_epochs": 5000,
    "symmetry_epochs": 1000,
    "hidden_dim": 16
  }
}
```

### Standalone Testing

Run the test script:

```bash
cd /home/ubuntu/LLM-SR/agentsr/src/tools/symmetry_discovery
./test_tool.sh
```

This will:
1. Create a test workspace
2. Generate/use test harmonic oscillator data
3. Run the symmetry discovery tool
4. Display results and compare with ground truth

### Manual Execution

```bash
# Setup environment
export WORKSPACE_ROOT="/path/to/workspace"
export WORKSPACE_INPUT="$WORKSPACE_ROOT/input"
export WORKSPACE_OUTPUT="$WORKSPACE_ROOT/output"
export WORKSPACE_LOGS="$WORKSPACE_ROOT/logs"
export WORKSPACE_SCRATCH="$WORKSPACE_ROOT/scratch"

# Set tool arguments
export TOOL_ARG_DATA_FILE="my_data.h5"
export TOOL_ARG_SURROGATE_EPOCHS="5000"
export TOOL_ARG_SYMMETRY_EPOCHS="1000"

# Run the tool
cd /home/ubuntu/LLM-SR/agentsr/src/tools/symmetry_discovery
./run.sh
```

## Configuration Parameters

### Surrogate Model

- `hidden_dim` (default: 16): Hidden layer dimension of MLP
- `surrogate_epochs` (default: 5000): Training epochs for surrogate
- `surrogate_lr` (default: 0.001): Learning rate for surrogate training
- `surrogate_batch_size` (default: 256): Batch size for surrogate training

### Symmetry Discovery

- `symmetry_epochs` (default: 1000): Optimization epochs for Lie generator
- `symmetry_lr` (default: 0.01): Learning rate for symmetry optimization
- `symmetry_batch_size` (default: 256): Batch size for symmetry optimization

### Model Management

- `load_existing_model` (default: false): Load existing surrogate from workspace_scratch

## Output

The tool writes a JSON result to `workspace_scratch/result.json`:

```json
{
  "tool_name": "symmetry_discovery",
  "result_type": "symmetry",
  "status": "success",
  "lie_generator": [
    [0.01, -0.99],
    [0.99, 0.02]
  ],
  "symmetry_loss": 0.00123,
  "num_dependent_vars": 2,
  "dependent_variables": ["x1", "x2"],
  "configuration": { ... }
}
```

### Interpreting Results

- **lie_generator**: The discovered Lie generator matrix
  - Shape: `(q, q)` where `q` = number of dependent variables
  - Represents infinitesimal symmetry transformation

- **symmetry_loss**: Final optimization loss
  - Close to 0: Strong symmetry found
  - \> 0.1: Weak or no symmetry
  - \> 1.0: Likely no meaningful symmetry

- **Symmetry types**:
  - Diagonal matrix: Scaling symmetry
  - Skew-symmetric: Rotational symmetry
  - General matrix: Linear combination symmetry

## Example: Harmonic Oscillator

The included test dataset represents a harmonic oscillator:
```
dx1/dt = x2
dx2/dt = -x1
```

This system has a rotational symmetry with ground truth Lie generator:
```
G = [[0, -1],
     [1,  0]]
```

This represents rotations in the (x1, x2) plane. Running the tool should recover a matrix close to this (up to normalization).

## Customization

### Implementing Prolongation

The `compute_prolongation()` method in `LieSymmetryDiscovery` class is an interface that should be customized for your specific ODE/PDE system.

Current location: `tool.py` line ~90

Example for first-order ODE `dx/dt = f(x)`:
```python
def compute_prolongation(self, x, derivatives):
    # Action on dependent variables
    v_x = (self.lie_generator @ x.T).T

    # Prolongation to first derivatives
    # v^(1) = Jacobian(v_x) @ f(x)
    # This requires computing Jacobian of v_x and evaluating f
    # Implementation depends on your specific system

    # Placeholder - implement your formula here
    v_derivatives = ...

    return torch.cat([v_x, v_derivatives], dim=1)
```

## Dependencies

- Python 3.8+
- PyTorch
- h5py
- numpy

Install via conda:
```bash
conda create -n pytorch python=3.10 pytorch h5py numpy -c pytorch -c conda-forge
conda activate pytorch
```

## Integration with Symbolic Regression

Discovered symmetries can guide symbolic regression:

1. **Reduce dimensionality**: Use symmetry to eliminate redundant variables
2. **Constrain search**: Equations should be invariant under the symmetry
3. **Guide templates**: Design expression_spec to respect symmetries
4. **Validate results**: Check if discovered equations preserve symmetries

### Workflow

```
Data → Symmetry Discovery → Analyze symmetry → Design SR template → Run SR → Verify
```

## Troubleshooting

### High Symmetry Loss

- Increase `surrogate_epochs` to improve surrogate fit
- Increase `symmetry_epochs` for better convergence
- Adjust learning rates
- Check if system actually has symmetries

### Poor Surrogate Fit

- Increase `hidden_dim` for more capacity
- Increase `surrogate_epochs`
- Adjust `surrogate_lr`
- Try different batch sizes

### Out of Memory

- Reduce `batch_size` parameters
- Reduce number of trajectories in dataset
- Use CPU instead of GPU for small datasets

## Future Extensions

- [ ] Non-linear Lie generators
- [ ] Symmetries on full space (independent + dependent variables)
- [ ] Multi-parameter Lie groups (multiple symmetries)
- [ ] Discrete symmetries (reflections, permutations)
- [ ] Automatic prolongation formula derivation
- [ ] Symmetry-constrained symbolic regression

## References

1. Olver, P. J. (1993). *Applications of Lie Groups to Differential Equations*. Springer.
2. Cranmer, M. et al. (2020). "Discovering Symbolic Models from Deep Learning with Inductive Biases". NeurIPS.
3. Noether's Theorem and conservation laws in physics.

## Contact

For questions or issues, please refer to the main agent-SR framework documentation.
