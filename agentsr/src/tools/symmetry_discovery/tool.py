#!/usr/bin/env python3
"""
Symmetry Discovery Tool - Discover linear continuous symmetries in ODE/PDE systems

This tool:
1. Loads HDF5 dataset with ODE/PDE data
2. Trains a surrogate neural network (MLP) to approximate the target function
3. Discovers linear continuous symmetries using Lie group theory
4. Returns the discovered symmetry generators and optimization loss

Environment Variables:
    WORKSPACE_INPUT: Path to workspace input directory
    WORKSPACE_OUTPUT: Path to workspace output directory
    WORKSPACE_LOGS: Path to workspace logs directory
    WORKSPACE_SCRATCH: Path to workspace scratch directory
    TOOL_ARG_*: Tool arguments from the LLM's tool_call JSON
"""

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from torch.func import jvp

# Add parent directory to path to import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.result_manager import write_result

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Maximum samples per epoch
MAX_SAMPLES_PER_EPOCH = 1000

def parse_env_arg(key, default=None, arg_type=str):
    """
    Parse an environment variable argument.

    Args:
        key: Environment variable name (without TOOL_ARG_ prefix)
        default: Default value if not present
        arg_type: Type to convert to (str, int, float, bool, list, dict)

    Returns:
        Parsed value or default
    """
    env_key = f"TOOL_ARG_{key.upper()}"
    value = os.environ.get(env_key)

    if value is None:
        return default

    try:
        if arg_type == bool:
            return value.lower() in ('true', '1', 'yes')
        elif arg_type in (list, dict):
            return json.loads(value)
        elif arg_type == int:
            return int(value)
        elif arg_type == float:
            return float(value)
        else:
            return value
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse {env_key}={value} as {arg_type.__name__}, using default: {default}", file=sys.stderr)
        return default


class ODEDataset(Dataset):
    """
    PyTorch dataset for ODE/PDE data from HDF5 files.

    The dataset flattens all trajectories and timesteps for random sampling.
    """

    def __init__(self, h5_path: str):
        """
        Initialize dataset from HDF5 file.

        Args:
            h5_path: Path to HDF5 file

        Expected HDF5 structure:
            - Each variable as a separate field (e.g., "u", "v", "x1", "x2")
            - "independent_variables": list of independent variable names (e.g., ["t"])
            - "dependent_variables": list of dependent variable names (e.g., ["x1", "x2"])
            - "feature_variables": list of feature variable names for SR
            - "target_variables": list of target variable names for SR
        """
        print(f"Loading dataset from: {h5_path}", file=sys.stderr)

        with h5py.File(h5_path, 'r') as f:
            # Read metadata
            self.independent_vars = json.loads(f.attrs['independent_variables'])
            self.dependent_vars = json.loads(f.attrs['dependent_variables'])
            self.feature_vars = json.loads(f.attrs['feature_variables'])
            self.target_vars = json.loads(f.attrs['target_variables'])

            print(f"Independent variables: {self.independent_vars}", file=sys.stderr)
            print(f"Dependent variables: {self.dependent_vars}", file=sys.stderr)
            print(f"Feature variables: {self.feature_vars}", file=sys.stderr)
            print(f"Target variables: {self.target_vars}", file=sys.stderr)

            # Load data for all variables
            self.data = {}
            for var in self.feature_vars + self.target_vars:
                if var not in f:
                    raise ValueError(f"Variable '{var}' not found in HDF5 file")
                self.data[var] = f[var][:]
                print(f"Loaded {var}: shape {self.data[var].shape}", file=sys.stderr)

        # Flatten all data
        self._flatten_data()

    def _identify_derivatives(self):
        """
        Identify derivative variables based on naming convention.

        Derivatives are named like "u_t" where u is a dependent variable
        and t is an independent variable.

        Returns:
            derivative_map: dict mapping derivative name to (var, indep_var)
        """
        derivative_map = {}

        for var_name in self.feature_vars + self.target_vars:
            # Check if this follows derivative naming convention
            for dep_var in self.dependent_vars:
                for indep_var in self.independent_vars:
                    deriv_name = f"{dep_var}_{indep_var}"
                    if var_name == deriv_name:
                        derivative_map[var_name] = (dep_var, indep_var)

        return derivative_map

    def _flatten_data(self):
        """Flatten all trajectories and timesteps into a single dataset."""
        # Get the shape of the first variable to determine flattening strategy
        first_var = self.feature_vars[0]
        data_shape = self.data[first_var].shape

        print(f"Original data shape: {data_shape}", file=sys.stderr)

        # Identify derivatives
        self.derivative_map = self._identify_derivatives()
        print(f"Identified derivatives: {list(self.derivative_map.keys())}", file=sys.stderr)

        # Flatten to (N_total_points,) for each variable
        flattened_data = {}
        for var_name, var_data in self.data.items():
            # Flatten all dimensions except potentially the last one (if it's a vector field)
            if var_data.ndim > 2:
                # If (N_trajs, N_timesteps, dim), flatten to (N_trajs * N_timesteps, dim)
                flattened = var_data.reshape(-1, var_data.shape[-1])
            else:
                # If (N_trajs, N_timesteps), flatten to (N_trajs * N_timesteps,)
                flattened = var_data.flatten()
            flattened_data[var_name] = flattened

        # Build structured data
        # Features for surrogate model
        X_list = []
        for var in self.feature_vars:
            data = flattened_data[var]
            if data.ndim == 1:
                X_list.append(data.reshape(-1, 1))
            else:
                X_list.append(data)
        self.X = np.hstack(X_list).astype(np.float32)

        # Targets for surrogate model
        y_list = []
        for var in self.target_vars:
            data = flattened_data[var]
            if data.ndim == 1:
                y_list.append(data.reshape(-1, 1))
            else:
                y_list.append(data)
        self.y = np.hstack(y_list).astype(np.float32)

        # Dependent variables (for symmetry)
        dependent_list = []
        for var in self.dependent_vars:
            if var in flattened_data:
                data = flattened_data[var]
                if data.ndim == 1:
                    dependent_list.append(data.reshape(-1, 1))
                else:
                    dependent_list.append(data)
            else:
                raise ValueError(f"Dependent variable '{var}' not found in data")
        self.dependent_data = np.hstack(dependent_list).astype(np.float32)

        # Derivative data (for prolongation)
        # Collect all derivatives that are in the target variables
        derivative_list = []
        self.derivative_names = []
        for target_var in self.target_vars:
            if target_var in self.derivative_map:
                data = flattened_data[target_var]
                if data.ndim == 1:
                    derivative_list.append(data.reshape(-1, 1))
                else:
                    derivative_list.append(data)
                self.derivative_names.append(target_var)

        if derivative_list:
            self.derivative_data = np.hstack(derivative_list).astype(np.float32)
        else:
            # No derivatives identified - use targets as fallback
            self.derivative_data = self.y.copy()
            self.derivative_names = self.target_vars

        print(f"Flattened X (features) shape: {self.X.shape}", file=sys.stderr)
        print(f"Flattened y (targets) shape: {self.y.shape}", file=sys.stderr)
        print(f"Dependent variables shape: {self.dependent_data.shape}", file=sys.stderr)
        print(f"Derivatives shape: {self.derivative_data.shape}", file=sys.stderr)
        print(f"Derivative names: {self.derivative_names}", file=sys.stderr)
        print(f"Total samples: {len(self.X)}", file=sys.stderr)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a structured dict with all relevant data for a sample.
        """
        return {
            'features': torch.from_numpy(self.X[idx]),
            'targets': torch.from_numpy(self.y[idx]),
            'dependent_vars': torch.from_numpy(self.dependent_data[idx]),
            'derivatives': torch.from_numpy(self.derivative_data[idx])
        }

    def get_feature_dim(self):
        return self.X.shape[1]

    def get_target_dim(self):
        return self.y.shape[1]
    
    def get_derivative_names(self):
        return self.derivative_names

    def get_num_dependent_vars(self):
        return len(self.dependent_vars)

    def get_num_derivatives(self):
        return len(self.derivative_names)


class SurrogateMLPRegressor(nn.Module):
    """
    Simple MLP surrogate model for function approximation.

    Maps from feature variables to target variables.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 16):
        """
        Initialize MLP.

        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
            hidden_dim: Hidden layer dimension (default: 16)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # 3-layer MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """Forward pass."""
        return self.net(x)


class LieSymmetryDiscovery:
    """
    Discover linear continuous symmetries using Lie group theory.

    This class learns a Lie generator (vector field) that represents
    an infinitesimal symmetry transformation of the ODE/PDE system.
    """

    def __init__(self, num_dependent_vars: int):
        """
        Initialize symmetry discovery.

        Args:
            num_dependent_vars: Number of dependent variables (dimension of state space)
        """
        self.num_dependent_vars = num_dependent_vars

        # Initialize Lie generator as a q x q matrix
        # This represents the infinitesimal generator of the symmetry transformation
        self.lie_generator = torch.randn(num_dependent_vars, num_dependent_vars,
                                         requires_grad=True, dtype=torch.float32)

        print(f"Initialized Lie generator with shape: {self.lie_generator.shape}", file=sys.stderr)

    def get_generator(self, normalize=True):
        eps = 1e-12
        frob = torch.linalg.norm(self.lie_generator)
        if normalize:
            return self.lie_generator / (frob + eps)
        return self.lie_generator

    def compute_prolongation(self, x: torch.Tensor, derivatives: torch.Tensor) -> torch.Tensor:
        """
        Compute the prolongation of the Lie generator to derivatives.

        Args:
            x: Dependent variables [batch_size, num_dependent_vars]
            derivatives: Derivatives (targets) [batch_size, num_derivatives]

        Returns:
            v: Infinitesimal symmetry transformation on all variables and derivatives
               Shape: [batch_size, num_dependent_vars + num_derivatives]
        """
        B, q = x.shape
        A = self.lie_generator

        v_x = x @ A.T  # [batch_size, num_dependent_vars]

        # Action on derivatives via prolongation
        if derivatives is None or derivatives.numel() == 0:
            v = v_x
            return v

        B2, m = derivatives.shape
        
        # Most common layout: derivatives are grouped by dependent variable, i.e. m = k * q.
        # Then reshape to [B, k, q] and apply the same A on the dependent-var axis.
        if m % q == 0:
            k = m // q
            deriv_blocks = derivatives.reshape(B, k, q)      # [B, k, q]
            v_deriv_blocks = deriv_blocks @ A.T             # [B, k, q]
            v_derivatives = v_deriv_blocks.reshape(B, m)    # [B, m]
        else:
            # Fallback: if the derivatives are not block-aligned, we cannot apply a principled
            # Olver prolongation without knowing how each column maps to a particular u^alpha_J.
            # We choose a conservative behavior: apply A only if m == q, otherwise raise error.
            if m == q:
                v_derivatives = derivatives @ A.T           # [B, q]
            else:
                raise ValueError("Derivatives not block-aligned.")

        # Concatenate to get full transformation
        v = torch.cat([v_x, v_derivatives], dim=1)

        return v

    def compute_symmetry_error(
        self,
        surrogate_model: nn.Module,
        batch_data: Dict[str, torch.Tensor],
        *,
        use_model_targets: bool = True,                # True: use y=f(x); False: use batch_data["targets"]
    ) -> torch.Tensor:
        """
        Symmetry error for a surrogate relation y = f(features), using Olver graph tangency:
            J_f(features) v_feat(features)  ==  v_tgt(features, y)

        Assumptions:
        - evolutionary/vertical symmetry: eta(u)=A u, xi=0
        - targets are concatenated full-jet blocks: targets.shape[1] % q == 0
        - prolongation on any jet block is A times that block
        """

        features = batch_data["features"]          # [B, in_dim]
        targets_true = batch_data["targets"]       # [B, out_dim]
        u = batch_data["dependent_vars"]           # [B, q]

        B, in_dim = features.shape
        q = self.num_dependent_vars
        A = self.get_generator()                   # [q, q]

        if in_dim % q != 0:
            raise ValueError(
                f"Feature dim must be a multiple of num_dependent_vars (q={q}) under "
                f"block-aligned jet assumption. Got in_dim={in_dim}."
            )

        k_feat = in_dim // q
        feat_blocks = features.reshape(B, k_feat, q)     # [B, k_feat, q]
        v_feat = (feat_blocks @ A.T).reshape(B, in_dim)  # [B, in_dim]

        # Batched JVP: returns (f(x), J_f(x) @ v_feat)
        y_pred, jac_v = jvp(surrogate_model, (features,), (v_feat,))   # both [B, out_dim]
        out_dim = y_pred.shape[1]
        # Choose y at which to evaluate v_tgt (graph tangency prefers y_pred)
        y_eval = y_pred if use_model_targets else targets_true
        if out_dim % q != 0:
            raise ValueError(
                f"Targets/output dim must be a multiple of num_dependent_vars (q={q}). "
                f"Got out_dim={out_dim}."
            )

        # Prolongation restricted to target coordinates:
        # targets are full jet blocks, so apply A to each block.
        k = out_dim // q
        y_blocks = y_eval.reshape(B, k, q)        # [B, k, q]
        v_tgt = (y_blocks @ A.T).reshape(B, out_dim)

        # Graph tangency residual
        residual = jac_v - v_tgt
        return torch.mean(residual ** 2)

    def optimize(self,
                 surrogate_model: nn.Module,
                 dataset: ODEDataset,
                 num_epochs: int = 1000,
                 batch_size: int = 256,
                 lr: float = 0.01,
                 device: str = 'cpu') -> Tuple[np.ndarray, float]:
        """
        Optimize the Lie generator to minimize symmetry error.

        Args:
            surrogate_model: Trained (frozen) surrogate model
            dataset: ODE dataset
            num_epochs: Number of optimization epochs
            batch_size: Batch size for optimization
            lr: Learning rate
            device: Device to run on

        Returns:
            lie_generator_matrix: Optimized Lie generator as numpy array
            final_loss: Final symmetry loss value
        """

        print("\n" + "="*60, file=sys.stderr)
        print("Optimizing Lie Generator for Symmetry Discovery", file=sys.stderr)
        print("="*60, file=sys.stderr)

        # Freeze surrogate model
        surrogate_model.eval()
        for param in surrogate_model.parameters():
            param.requires_grad = False

        # Move to device - need to preserve leaf status for optimization
        surrogate_model = surrogate_model.to(device)

        self.lie_generator = nn.Parameter(self.lie_generator.detach().to(device))

        # Create dataloader with limited samples per epoch
        num_samples = min(len(dataset), MAX_SAMPLES_PER_EPOCH)
        sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        print(f"Using {num_samples} samples per epoch (max {MAX_SAMPLES_PER_EPOCH})", file=sys.stderr)

        # Optimizer for Lie generator
        optimizer = optim.Adam([self.lie_generator], lr=lr)

        # Optimization loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_data in dataloader:
                # Move all data to device
                batch_data = {k: v.to(device) for k, v in batch_data.items()}

                optimizer.zero_grad()

                # Compute symmetry error
                loss = self.compute_symmetry_error(surrogate_model, batch_data)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Symmetry Loss: {avg_loss:.6f}",
                      file=sys.stderr)

        final_loss = avg_loss

        print("="*60, file=sys.stderr)
        print(f"Optimization complete. Final loss: {final_loss:.6f}", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)

        # Convert to numpy
        lie_generator_matrix = self.get_generator().detach().cpu().numpy()

        return lie_generator_matrix, final_loss


def train_surrogate_model(dataset: ODEDataset,
                          hidden_dim: int = 16,
                          num_epochs: int = 5000,
                          batch_size: int = 256,
                          lr: float = 0.001,
                          device: str = 'cpu',
                          model_save_path: Optional[str] = None,
                          load_existing: bool = False) -> SurrogateMLPRegressor:
    """
    Train a surrogate MLP model.

    Args:
        dataset: ODE dataset
        hidden_dim: Hidden dimension of MLP
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to run on
        model_save_path: Path to save/load model weights
        load_existing: Whether to try loading existing weights

    Returns:
        Trained surrogate model
    """

    input_dim = dataset.get_feature_dim()
    output_dim = dataset.get_target_dim()

    print("\n" + "="*60, file=sys.stderr)
    print("Training Surrogate MLP Model", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Input dimension: {input_dim}", file=sys.stderr)
    print(f"Output dimension: {output_dim}", file=sys.stderr)
    print(f"Hidden dimension: {hidden_dim}", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    # Initialize model
    model = SurrogateMLPRegressor(input_dim, output_dim, hidden_dim)
    model = model.to(device)

    # Try to load existing weights
    if load_existing and model_save_path and os.path.exists(model_save_path):
        print(f"Loading existing model from: {model_save_path}", file=sys.stderr)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("Model loaded successfully!", file=sys.stderr)
        return model

    # Create dataloader with limited samples per epoch
    num_samples = min(len(dataset), MAX_SAMPLES_PER_EPOCH)
    sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print(f"Using {num_samples} samples per epoch (max {MAX_SAMPLES_PER_EPOCH})", file=sys.stderr)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in dataloader:
            # Extract features and targets
            batch_x = batch_data['features'].to(device)
            batch_y = batch_data['targets'].to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], MSE Loss: {avg_loss:.6f}",
                  file=sys.stderr)

    print("="*60, file=sys.stderr)
    print(f"Training complete. Final MSE: {avg_loss:.6f}", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    # Save model
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}", file=sys.stderr)

    return model, avg_loss


def main():
    """Main execution function."""
    try:
        # Get HDF5 data file from environment
        data_file = os.environ.get('TOOL_ARG_DATA_FILE')
        if not data_file:
            raise ValueError("TOOL_ARG_DATA_FILE environment variable not set")

        # Resolve path - could be absolute or relative to WORKSPACE_INPUT
        if not os.path.isabs(data_file):
            workspace_input = os.environ.get('WORKSPACE_INPUT', '')
            data_file = os.path.join(workspace_input, data_file)

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        print(f"Using data file: {data_file}", file=sys.stderr)

        # Parse configuration
        hidden_dim = parse_env_arg('hidden_dim', 16, int)
        surrogate_epochs = parse_env_arg('surrogate_epochs', 500, int)
        surrogate_lr = parse_env_arg('surrogate_lr', 0.001, float)
        surrogate_batch_size = parse_env_arg('surrogate_batch_size', 256, int)

        symmetry_epochs = parse_env_arg('symmetry_epochs', 500, int)
        symmetry_lr = parse_env_arg('symmetry_lr', 0.01, float)
        symmetry_batch_size = parse_env_arg('symmetry_batch_size', 256, int)

        load_existing_model = parse_env_arg('load_existing_model', False, bool)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}", file=sys.stderr)

        # Model save path
        workspace_scratch = os.environ.get('WORKSPACE_SCRATCH', './workspace_scratch')
        model_save_path = os.path.join(workspace_scratch, 'surrogate_model.pth')

        # Load dataset
        dataset = ODEDataset(data_file)

        # Train surrogate model
        surrogate_model, predictor_loss = train_surrogate_model(
            dataset=dataset,
            hidden_dim=hidden_dim,
            num_epochs=surrogate_epochs,
            batch_size=surrogate_batch_size,
            lr=surrogate_lr,
            device=device,
            model_save_path=model_save_path,
            load_existing=load_existing_model
        )

        # Discover symmetry
        symmetry_discovery = LieSymmetryDiscovery(
            num_dependent_vars=dataset.get_num_dependent_vars()
        )

        lie_generator, symmetry_loss = symmetry_discovery.optimize(
            surrogate_model=surrogate_model,
            dataset=dataset,
            num_epochs=symmetry_epochs,
            batch_size=symmetry_batch_size,
            lr=symmetry_lr,
            device=device
        )

        # Prepare results
        results = {
            "tool_name": "symmetry_discovery",
            "result_type": "symmetry",
            "status": "success",
            "lie_generator": lie_generator.tolist(),
            "predictor_loss": round(float(predictor_loss), 6),
            "symmetry_loss": round(float(symmetry_loss), 6),
            # "num_dependent_vars": dataset.get_num_dependent_vars(),
            # "dependent_variables": dataset.dependent_vars,
            # "configuration": {
            #     "hidden_dim": hidden_dim,
            #     "surrogate_epochs": surrogate_epochs,
            #     "surrogate_lr": surrogate_lr,
            #     "symmetry_epochs": symmetry_epochs,
            #     "symmetry_lr": symmetry_lr
            # }
        }

        # Write results
        result_path = write_result(results, tool_name="symmetry_discovery")
        print(f"\nResults written to: {result_path}", file=sys.stderr)

        print("\n" + "="*60, file=sys.stderr)
        print("SYMMETRY DISCOVERY COMPLETE", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"Lie Generator Matrix ({lie_generator.shape[0]}x{lie_generator.shape[1]}):",
              file=sys.stderr)
        print(lie_generator, file=sys.stderr)
        print(f"\nSymmetry Loss: {symmetry_loss:.6f}", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)

        return 0

    except Exception as e:
        # Output error as JSON to file
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

        # Write error to result file
        try:
            result_path = write_result(error_result, tool_name="symmetry_discovery")
            print(f"Error written to: {result_path}", file=sys.stderr)
        except Exception as write_error:
            print(f"Failed to write error to file: {write_error}", file=sys.stderr)

        print(f"\nError details: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
