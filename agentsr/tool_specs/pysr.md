#### Description

**pysr** uses evolutionary algorithms to discover mathematical equations that fit your data.

**Best For:**
- Complex non-linear relationships
- High-dimensional feature spaces
- When accuracy and expressiveness are priorities

**Strengths:**
- Evolutionary search through equation space
- Highly customizable operators (arithmetic, trigonometric, special functions)
- Multi-population parallel search
- Built-in simplification and regularization

**Limitations:**
- Computationally intensive for large equation searches
- May require parameter tuning for optimal performance

#### Required Arguments

- **`input_file`** (string): name of the data file under the input subdirectory of the workspace. Only include the file name instead of the full path (e.g., `data.csv` instead of `input/data.csv`).

- **`binary_operators`** (list of strings)
  - Binary operators (two inputs) to use in equation building
  - Common operators: `["+", "-", "*", "/"]`
  - Additional options: `"^"`, `"max"`, `"min"`, `"mod"`
  - Can define custom operators with Julia syntax: `"my_op(x, y) = x^2 + y^2"`
  - **Default:** `["+", "-", "*", "/"]`

- **`unary_operators`** (list of strings)
  - Unary operators (single input) to include in search
  - Common options: `["sin", "cos", "exp", "log", "sqrt", "abs"]`
  - Custom: `"square(x) = x^2"`, `"cube(x) = x^3"`
  - **Default:** `[]` (no unary operators)

#### Optional Arguments

##### Search Configuration

- **`niterations`** (int)
  - Number of iterations to run the evolutionary search
  - Each iteration performs multiple generations of evolution
  - Higher values → more thorough search, longer runtime
  - **Default:** `10`
  - **Typical range:** `5-40` (5 for quick tests, 20+ for production)

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

##### Constraints

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
  "tool_name": "pysr",
  "args": {
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": [],
    "niterations": 10,
    "maxsize": 15
  }
}
```

##### Non-linear with Trigonometric and Exponential Functions
```json
{
  "tool_name": "pysr",
  "args": {
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "exp", "log"],
    "niterations": 20,
    "maxsize": 20
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

3. **For periodic/oscillatory data:**
   - Include `["sin", "cos"]` in `unary_operators`
   - Consider adding `"tan"` if appropriate
   - May need higher `maxsize` for complex waveforms

4. **For exponential growth/decay:**
   - Include `["exp", "log"]` in `unary_operators`
   - Watch for numerical instability with large exponents
