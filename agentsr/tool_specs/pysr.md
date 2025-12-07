### PySR (Python Symbolic Regression)

#### Description

PySR is a high-performance symbolic regression tool that uses evolutionary algorithms to discover mathematical equations that fit your data. It is built on top of SymbolicRegression.jl (Julia) for computational efficiency while providing a Python interface compatible with scikit-learn.

**Main Interface:** `PySRRegressor` (for regression tasks)

**Best For:**
- Complex non-linear relationships
- Discovering interpretable mathematical equations from data
- High-dimensional feature spaces
- When accuracy and expressiveness are priorities

**Strengths:**
- Evolutionary search through equation space
- Highly customizable operators (arithmetic, trigonometric, special functions)
- Multi-population parallel search
- Built-in simplification and regularization
- Export to multiple formats (SymPy, PyTorch, JAX, C, etc.)

**Limitations:**
- Computationally intensive for large equation searches
- May require parameter tuning for optimal performance

#### Required Arguments

- **`binary_operators`** (list of strings)
  - Binary operators (two inputs) to use in equation building
  - Common operators: `["+", "-", "*", "/"]`
  - Additional options: `"^"`, `"max"`, `"min"`, `"mod"`
  - Can define custom operators with Julia syntax: `"my_op(x, y) = x^2 + y^2"`
  - **Default:** `["+", "-", "*", "/"]`

- **`unary_operators`** (list of strings)
  - Unary operators (single input) to include in search
  - Common options: `["sin", "cos", "exp", "log", "sqrt", "abs"]`
  - Advanced: `"tanh"`, `"sinh"`, `"cosh"`, `"erf"`, `"gamma"`
  - Custom: `"square(x) = x^2"`, `"cube(x) = x^3"`
  - **Default:** `[]` (no unary operators)

#### Optional Arguments

##### Search Configuration

- **`niterations`** (int)
  - Number of iterations to run the evolutionary search
  - Each iteration performs multiple generations of evolution
  - Higher values → more thorough search, longer runtime
  - **Default:** `40`
  - **Typical range:** `5-100` (5 for quick tests, 40+ for production)

- **`populations`** (int)
  - Number of independent populations to evolve in parallel
  - Populations periodically exchange best equations (migration)
  - More populations → better exploration but more computation
  - **Default:** `15`
  - **Typical range:** `10-30`

- **`population_size`** (int)
  - Number of equations in each population
  - Larger populations → better diversity, slower per-iteration
  - **Default:** `33`
  - **Typical range:** `20-100`

- **`ncycles_per_iteration`** (int)
  - Number of evolutionary cycles between migrations
  - Controls how often populations exchange information
  - **Default:** `550`
  - **Typical range:** `100-1000`

##### Complexity Control

- **`maxsize`** (int)
  - Maximum complexity (number of nodes) allowed in equations
  - Limits equation size to prevent overly complex models
  - Complexity = operators + features + constants
  - Example: `sin(x1) + x2 * 2.5` has complexity 5
  - **Default:** `20`
  - **Typical range:** `10-30` (start smaller, increase if needed)

- **`maxdepth`** (int or None)
  - Maximum nesting depth of expressions
  - Example: `sin(cos(x))` has depth 2
  - `None` means no depth limit (only maxsize applies)
  - **Default:** `None`
  - **Typical range:** `3-10` when used

##### Performance and Parallelism

- **`procs`** (int or None)
  - Number of processes to use for parallelization
  - `None` uses all available CPU cores
  - Set to `0` to disable parallelism
  - **Default:** `cpu_count()`

- **`multithreading`** (bool)
  - Use multithreading within Julia instead of multiprocessing
  - Can be faster for small populations
  - **Default:** `True`

- **`timeout_in_seconds`** (float or None)
  - Maximum time (in seconds) to run the search
  - Search stops early if timeout is reached
  - Useful for enforcing time budgets
  - **Default:** `None` (no timeout)

##### Feature Selection

- **`select_k_features`** (int or None)
  - Pre-select k most important features using random forest
  - Reduces dimensionality before symbolic regression
  - `None` means use all features
  - **Default:** `None`
  - **Use when:** You have many features (10+) and suspect some are irrelevant

##### Optimization and Refinement

- **`optimize_hof`** (bool)
  - Optimize constants in the Hall of Fame (best equations) using gradient descent
  - Improves equation accuracy through local optimization
  - **Default:** `True`

- **`warm_start`** (bool)
  - Continue search from previous `.fit()` call
  - Useful for incremental searches or parameter tuning
  - **Default:** `False`

##### Loss and Constraints

- **`loss`** (str)
  - Loss function to optimize
  - Options: `"L2DistLoss()"` (MSE), `"L1DistLoss()` (MAE), custom Julia expressions
  - **Default:** `"L2DistLoss()"` (mean squared error)

- **`constraints`** (dict or None)
  - Complexity constraints for specific operators
  - Example: `{"+": (5, 5), "*": (3, 3)}` limits argument complexity
  - Format: `{operator: (max_left_complexity, max_right_complexity)}`
  - **Default:** `None`

- **`nested_constraints`** (dict or None)
  - Constraints on operator nesting
  - Example: `{"sin": ["sin", "cos"]}` prevents `sin(sin(x))` and `sin(cos(x))`
  - **Default:** `None`

##### Noise Handling

- **`denoise`** (bool)
  - Use Gaussian Process to denoise data before fitting
  - Helps when data has significant noise
  - **Default:** `False`

##### Output and Interpretation

- **`extra_sympy_mappings`** (dict or None)
  - Map custom operators to SymPy equivalents
  - Required if you use custom operators and want SymPy export
  - Example: `{"inv": lambda x: 1/x}`
  - **Default:** `None`

#### Typical Usage Examples

##### Simple Linear/Polynomial Search
```json
{
  "tool_name": "PySR",
  "arguments": {
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": [],
    "niterations": 20,
    "populations": 15,
    "population_size": 33,
    "maxsize": 15
  }
}
```

##### Non-linear with Trigonometric Functions
```json
{
  "tool_name": "PySR",
  "arguments": {
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "exp", "log"],
    "niterations": 40,
    "populations": 20,
    "population_size": 50,
    "maxsize": 20
  }
}
```

##### Quick Exploration (Fast)
```json
{
  "tool_name": "PySR",
  "arguments": {
    "binary_operators": ["+", "-", "*"],
    "unary_operators": ["square"],
    "niterations": 5,
    "populations": 10,
    "population_size": 20,
    "maxsize": 10,
    "timeout_in_seconds": 60
  }
}
```

##### High-Dimensional with Feature Selection
```json
{
  "tool_name": "PySR",
  "arguments": {
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["exp", "log"],
    "select_k_features": 5,
    "niterations": 30,
    "populations": 15,
    "population_size": 40,
    "maxsize": 18
  }
}
```

#### Parameter Selection Guidelines

1. **For noisy data:**
   - Use simpler operators initially
   - Set `denoise: true`
   - Lower `maxsize` (10-15) to avoid overfitting

2. **For clean data:**
   - Can use higher `maxsize` (20-30)
   - Include more sophisticated operators (trigonometric, exponential)
   - Higher `niterations` for thorough search

3. **For time-constrained searches:**
   - Set `timeout_in_seconds`
   - Reduce `niterations` and `populations`
   - Start with simpler operator sets

4. **For periodic/oscillatory data:**
   - Include `["sin", "cos"]` in `unary_operators`
   - Consider adding `"tan"` if appropriate
   - May need higher `maxsize` for complex waveforms

5. **For exponential growth/decay:**
   - Include `["exp", "log"]` in `unary_operators`
   - Watch for numerical instability with large exponents
