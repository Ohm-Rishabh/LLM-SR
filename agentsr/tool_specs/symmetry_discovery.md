#### Description

**symmetry_discovery** discovers linear continuous symmetries in ODE/PDE systems using Lie group theory and neural network surrogates.

**Best For:**
- ODE/PDE systems where symmetries can provide insights into the governing equations
- Discovering invariances and conservation laws
- Reducing dimensionality of symbolic regression problems via symmetry constraints
- Systems with known or suspected physical symmetries (rotations, translations, scaling, etc.)

**Strengths:**
- Automatically discovers continuous symmetries without prior specification
- Uses Lie group theory to find infinitesimal generators
- Learns from data via neural network surrogate
- Can reveal hidden structure in the equations
- Symmetries can guide subsequent symbolic regression

**Limitations:**
- Currently limited to linear Lie generators (vector fields)
- Requires implementation of prolongation formula (provided as interface)
- Assumes differentiable surrogate model
- May require tuning of optimization hyperparameters
- Limited to symmetries on dependent variables (can be extended to include independent variables)

**How It Works:**
1. Loads ODE/PDE data from HDF5 file
2. Trains a 3-layer MLP surrogate to approximate the target function (derivatives)
3. Parameterizes a Lie generator as a matrix on the space of dependent variables
4. Optimizes the generator to minimize symmetry error: ||Jacobian(f) · v(x)||²
5. Returns the discovered Lie generator matrix and symmetry loss

#### Required Arguments

- **`data_file`** (string): Path to HDF5 data file (can be absolute or relative to workspace input directory)
  - Must contain fields for each variable as separate datasets
  - Must include attributes:
    - `independent_variables`: list of independent variable names (e.g., ["t"])
    - `dependent_variables`: list of dependent variable names (e.g., ["x1", "x2"])
    - `feature_variables`: list of variables used as features for SR
    - `target_variables`: list of variables used as targets for SR (e.g., derivatives)
  - Example: For a 2D ODE system with variables x1, x2 and their time derivatives:
    - Variables: "x1", "x2", "x1_t", "x2_t" (as separate datasets)
    - Attributes:
      - independent_variables = ["t"]
      - dependent_variables = ["x1", "x2"]
      - feature_variables = ["x1", "x2"]
      - target_variables = ["x1_t", "x2_t"]

#### Optional Arguments

##### Surrogate Model Configuration

- **`hidden_dim`** (int)
  - Hidden layer dimension for the 3-layer MLP surrogate
  - Larger values increase model capacity but may overfit
  - **Default:** `16`
  - **Typical range:** `8-64`

- **`surrogate_epochs`** (int)
  - Number of training epochs for surrogate model
  - More epochs improve fit but increase training time
  - **Default:** `5000`
  - **Typical range:** `1000-10000`

- **`surrogate_lr`** (float)
  - Learning rate for surrogate model training
  - **Default:** `0.001`
  - **Typical range:** `0.0001-0.01`

- **`surrogate_batch_size`** (int)
  - Batch size for surrogate training
  - **Default:** `256`
  - **Typical range:** `64-1024`

##### Symmetry Discovery Configuration

- **`symmetry_epochs`** (int)
  - Number of optimization epochs for Lie generator
  - More epochs allow better convergence to symmetry
  - **Default:** `1000`
  - **Typical range:** `500-5000`

- **`symmetry_lr`** (float)
  - Learning rate for Lie generator optimization
  - **Default:** `0.01`
  - **Typical range:** `0.001-0.1`

- **`symmetry_batch_size`** (int)
  - Batch size for symmetry optimization
  - **Default:** `256`
  - **Typical range:** `64-1024`

##### Model Management

- **`load_existing_model`** (bool)
  - Whether to load existing surrogate model weights from workspace_scratch
  - If True and model exists, skips surrogate training
  - If False or model doesn't exist, trains new surrogate
  - **Default:** `false`
  - Model is automatically saved to `workspace_scratch/surrogate_model.pth`

#### Output Format

The tool returns a JSON result with the following structure:

```json
{
  "tool_name": "symmetry_discovery",
  "result_type": "symmetry",
  "status": "success",
  "lie_generator": [[...], [...], ...],
  "symmetry_loss": 0.00123,
  "num_dependent_vars": 2,
  "dependent_variables": ["x1", "x2"],
  "configuration": {
    "hidden_dim": 16,
    "surrogate_epochs": 5000,
    "surrogate_lr": 0.001,
    "symmetry_epochs": 1000,
    "symmetry_lr": 0.01
  }
}
```

**Key Fields:**
- `lie_generator`: The discovered Lie generator matrix (q × q where q = number of dependent variables)
- `symmetry_loss`: Final symmetry error (lower is better, close to 0 indicates strong symmetry)
- `num_dependent_vars`: Dimension of the state space
- `dependent_variables`: Names of the dependent variables

#### Usage Examples

**IMPORTANT: Do NOT include comments in the JSON structure.**

##### Example 1: Simple 2D ODE system

```json
{
  "tool_name": "symmetry_discovery",
  "args": {
    "data_file": "harmonic_oscillator.h5",
    "surrogate_epochs": 3000,
    "symmetry_epochs": 1000
  }
}
```

##### Example 2: Higher-dimensional system with custom settings

```json
{
  "tool_name": "symmetry_discovery",
  "args": {
    "data_file": "lotka_volterra.h5",
    "hidden_dim": 32,
    "surrogate_epochs": 5000,
    "surrogate_lr": 0.0005,
    "symmetry_epochs": 2000,
    "symmetry_lr": 0.01,
    "surrogate_batch_size": 512
  }
}
```

##### Example 3: Using existing surrogate model

```json
{
  "tool_name": "symmetry_discovery",
  "args": {
    "data_file": "pendulum.h5",
    "load_existing_model": true,
    "symmetry_epochs": 1500
  }
}
```

#### Interpreting Results

**Lie Generator Matrix:**
- Represents the infinitesimal generator of a one-parameter Lie group
- Each row corresponds to a dependent variable
- Describes how variables transform under the symmetry
- Can be exponentiated to get finite transformations: exp(ε·G) for small ε

**Symmetry Loss:**
- Measures how well the discovered transformation preserves the equations
- Loss ≈ 0: Strong symmetry found
- Loss > 0.1: Weak or no symmetry (may need more training or different hyperparameters)
- Loss > 1.0: Likely no meaningful symmetry

**Common Symmetries in Physical Systems:**
- **Time translation:** Variables shift uniformly (G diagonal)
- **Scaling symmetry:** Variables scale together (G proportional to identity)
- **Rotational symmetry:** Pairs of variables rotate (G skew-symmetric blocks)
- **Linear combination:** Variables mix linearly (general G)

#### Integration with Symbolic Regression

Discovered symmetries can be used to:
1. **Reduce search space:** Use symmetry to eliminate redundant variables
2. **Constrain equation form:** Equations should be invariant under the symmetry
3. **Guide template design:** Structure templates to respect symmetries
4. **Validate results:** Check if discovered equations preserve the symmetry

**Workflow:**
1. Run `symmetry_discovery` first to find Lie generators
2. Analyze the generator to understand the symmetry type
3. Design symbolic regression templates that respect the symmetry
4. Run symbolic regression with symmetry-informed constraints

#### Implementation Notes

**Prolongation Formula:**
The tool provides an interface `compute_prolongation()` for extending the Lie generator to derivatives. The current implementation uses a placeholder that should be replaced with the specific prolongation formula for your system.

For a first-order ODE system dx/dt = f(t, x):
- The prolongation extends the generator from x-space to (x, dx/dt)-space
- Formula depends on the structure of the ODE
- Common case: v^(1) = D_t(v) + (∂f/∂x)·v where D_t is total derivative

**Future Extensions:**
- Support for non-linear Lie generators
- Symmetries on the full space (independent + dependent variables)
- Multi-parameter Lie groups (multiple symmetries simultaneously)
- Discrete symmetries (reflections, permutations)

#### Technical Requirements

**Python Dependencies:**
- torch (PyTorch)
- h5py
- numpy

**Conda Environment:**
The tool attempts to activate a conda environment named `pytorch`. Ensure you have:
```bash
conda create -n pytorch python=3.10 pytorch h5py -c pytorch -c conda-forge
conda activate pytorch
```

**Data Preparation:**
Use h5py to create HDF5 files:
```python
import h5py
import numpy as np

with h5py.File('data.h5', 'w') as f:
    # Store variables
    f.create_dataset('x1', data=x1_array)
    f.create_dataset('x2', data=x2_array)
    f.create_dataset('x1_t', data=x1_t_array)
    f.create_dataset('x2_t', data=x2_t_array)

    # Store metadata
    f.attrs['independent_variables'] = json.dumps(['t'])
    f.attrs['dependent_variables'] = json.dumps(['x1', 'x2'])
    f.attrs['feature_variables'] = json.dumps(['x1', 'x2'])
    f.attrs['target_variables'] = json.dumps(['x1_t', 'x2_t'])
```
