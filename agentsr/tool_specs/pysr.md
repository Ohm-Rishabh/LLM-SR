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
- Template-based search for known equation structures (HIGHLY RECOMMENDED)

**Limitations:**
- Computationally intensive for large equation searches
- May require parameter tuning for optimal performance

**IMPORTANT - Use Template Expressions When Possible:**
Using `expression_spec` to specify equation structure is STRONGLY RECOMMENDED as it:
- Dramatically reduces search space and runtime (often 10-100x faster)
- Produces better results by constraining search to plausible forms
- Allows you to leverage domain knowledge from pretraining

**Critical: Make educated guesses from domain knowledge:**
- Draw on your understanding of the physical system, domain, or context from pretraining
- Consider results from previous tool calls (visualizations, statistical analysis, data exploration)
- Think about the governing principles (conservation laws, scaling relationships, rate equations, etc.)
- Reason about what functional forms are plausible for the specific phenomenon
- DO NOT blindly apply generic templates - tailor the structure to the specific problem

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

##### Template Expression Specification (HIGHLY RECOMMENDED)

- **`expression_spec`** (dict or None)
  - **USE THIS WHENEVER POSSIBLE** to dramatically speed up search and improve results
  - Specifies the structure/template of the equation you're searching for based on domain reasoning
  - Allows breaking complex equations into smaller sub-expressions that are learned separately
  - Format: `{"expressions": [...], "variable_names": [...], "combine": "..."}`
  - **Default:** `None` (unrestricted search - much slower)

  **CRITICAL - Understanding the Three Arguments:**

  **1. `expressions`: PLACEHOLDER function names** (what PySR will discover)
  - ✅ **CORRECT**: `["f", "g", "h"]` - symbolic names representing unknown sub-expressions
  - ✅ **CORRECT**: `["func1", "func2"]` - any simple symbolic names
  - ❌ **WRONG**: `["F", "a", "m"]` - DO NOT use actual variable/column names
  - ❌ **WRONG**: `["x1", "x2"]` - DO NOT confuse with feature names
  - ❌ **WRONG**: `["F/a"]` - DO NOT use expressions
  - Think of these as "blanks" that PySR will fill in

  **2. `variable_names`: FEATURE column names** (from your CSV data)
  - ✅ **CORRECT**: `["F", "a"]` - if your CSV has columns F, a, m and you're predicting m
  - ✅ **CORRECT**: `["velocity", "mass", "force"]` - actual column names
  - ❌ **WRONG**: Including the target variable (don't include the column you're trying to predict)
  - ❌ **WRONG**: `["f", "g"]` - these are placeholder names, not data columns
  - Must match actual column names in your input CSV (excluding the target)

  **3. `combine`: Template string** (how placeholders and known parts combine)
  - **BE AS SPECIFIC AS YOUR DOMAIN KNOWLEDGE ALLOWS**
  - If you know certain operations/functions appear, include them explicitly
  - Only use placeholders for the unknown parts

  Examples ranked from most to least specific:
  - ✅ **BEST (very specific)**: `"F / a + f(F, a)"` - if you strongly believe in F/a plus a correction
  - ✅ **GOOD (moderately specific)**: `"sin(f(x1)) * g(x2, x3)"` - know it involves sin, but not the internals
  - ✅ **OKAY (generic structure)**: `"f(x1, x2) + g(x3)"` - only know additive structure
  - ❌ **WRONG**: `"f(x1, x2, x3)"` - too generic, just use unrestricted search instead
  - ❌ **WRONG**: `"f(m, F, a)"` - don't include target variable `m`

  **Key principle**: More specificity = faster search. Include any known functions (sin, exp, sqrt, division, etc.) directly in the combine string, only use placeholders for truly unknown sub-expressions.

  **Example of CORRECT usage:**
  ```json
  {
    "expressions": ["f", "g"],           // Placeholders PySR will discover
    "variable_names": ["F", "a"],        // Feature columns from CSV (not target)
    "combine": "f(F) * g(a)"            // How f and g combine using F and a
  }
  ```

  **Common MISTAKES to avoid:**
  - Using data column names in `expressions` list
  - Using placeholder names in `variable_names` list
  - Including target variable in `variable_names`
  - Putting a complete equation (with no placeholders) in `combine`

  **How to design the template (think critically, don't copy examples):**

  1. **Analyze the physical/domain context:**
     - What physical principles govern this system? (Newton's laws, thermodynamics, chemical kinetics, etc.)
     - What are the units and dimensions? Can dimensional analysis suggest the form?
     - Are there known scaling laws or invariances in this domain?
     - What prior knowledge do you have about similar systems from pretraining?

  2. **Use insights from previous tool calls:**
     - What did visualizations reveal? (linear trends, exponential growth, periodic behavior, etc.)
     - What did correlation analysis show about variable relationships?
     - Are there outliers or regime changes that suggest piecewise or composite structures?
     - Did statistical analysis suggest separable contributions?

  3. **Reason about functional composition:**
     - Should components combine additively (independent contributions) or multiplicatively (coupled effects)?
     - Are there nested transformations? (e.g., response to a transformed input like exp(rate*time))
     - Are certain variables only relevant within specific functions? (enables decomposition)
     - Do symmetries or conservation laws constrain the form?

  4. **Common domain-specific patterns to consider:**
     - **Mechanics/dynamics**: Look for force balances, energy terms, damping factors
     - **Thermodynamics**: Look for exponential Boltzmann factors, power law dependencies
     - **Chemical kinetics**: Look for rate equations, Arrhenius terms, mass action law
     - **Fluid dynamics**: Look for Reynolds number dependencies, boundary layer terms
     - **Electrical systems**: Look for RC/RL time constants, resonance terms
     - **Biology**: Look for Michaelis-Menten kinetics, logistic growth, allometric scaling
     - **Economics**: Look for elasticities, growth rates, equilibrium conditions

  **Syntax examples (for reference only - DO NOT blindly copy these structures):**
  - Additive: `"f(x1, x2) + g(x3)"` - use when contributions are independent
  - Multiplicative: `"f(x1) * g(x2, x3)"` - use when effects are coupled
  - Nested: `"sin(f(x1, x2))"` or `"exp(f(x1))"` - use when transformations are expected
  - Mixed: `"f(x1) * exp(g(x2)) + h(x3)"` - use when domain knowledge suggests specific combinations

##### Search Configuration

- **`niterations`** (int)
  - Number of iterations to run the evolutionary search
  - Each iteration performs multiple generations of evolution
  - Higher values → more thorough search, longer runtime
  - **Default:** `10`
  - **Typical range:** `5-40` (5 for quick tests, 20+ for production)
  - **With templates:** Can often use lower values (5-15) since search space is reduced

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

#### Usage Examples

**CRITICAL: These are illustrative JSON structures only. You MUST design your own `expression_spec` based on domain reasoning, NOT copy these examples.**

##### Example: Using a template (when justified by domain analysis)
```json
{
  "tool_name": "pysr",
  "args": {
    "input_file": "data.csv",
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": [],
    "expression_spec": {
      "expressions": ["f", "g"],
      "variable_names": ["x1", "x2", "x3"],
      "combine": "<YOUR_DOMAIN_SPECIFIC_STRUCTURE>"
    },
    "niterations": 10,
    "maxsize": 12
  }
}
```

Where `<YOUR_DOMAIN_SPECIFIC_STRUCTURE>` should be replaced with your reasoned hypothesis, such as:
- For separable physics: maybe `"f(x1, x2) * g(x3)"` if domain suggests coupled multiplicative effects
- For chemical kinetics: maybe `"f(x1) * exp(-g(x2) / x3)"` if you recognize Arrhenius-like behavior
- For oscillatory systems: maybe `"f(x1) * sin(g(x2) + h(x3))"` if phase and amplitude are separable
- DO NOT use these literally - they are domain-specific examples for illustration

##### Fallback: Unrestricted search (only when truly no structural knowledge exists)
Use this ONLY when you genuinely have no domain knowledge AND previous tool calls revealed no patterns:
```json
{
  "tool_name": "pysr",
  "args": {
    "input_file": "data.csv",
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": [],
    "niterations": 10,
    "maxsize": 15
  }
}
```

#### Parameter Selection Guidelines

**REASONING WORKFLOW - Follow this process:**

1. **First, engage in domain reasoning (MOST IMPORTANT):**
   - What physical/domain context is this? What governing equations or principles apply?
   - Review results from previous tool calls - what patterns emerged from visualization/analysis?
   - Can you make an educated hypothesis about the functional form?
   - If you can hypothesize a structure → Design a custom `expression_spec` for 10-100x speedup

2. **Design operators based on your hypothesis:**
   - **AVOID adding operators "just in case"** - each operator exponentially increases search space
   - Include `sin`, `cos` ONLY if you have evidence of periodicity or domain knowledge suggests it
   - Include `exp`, `log` ONLY if you see exponential trends or domain knowledge suggests it
   - For most cases, start with just `["+", "-", "*", "/"]` and no unary operators
   - Add complexity only when justified by domain reasoning or previous observations

3. **Set complexity parameters based on template usage:**
   - **With templates:** Use smaller `maxsize` (8-12) since sub-expressions are simpler
   - **With templates:** Reduce `niterations` (5-15) since search space is constrained
   - **Without templates:** May need larger `maxsize` (15-25) but still avoid unnecessarily large values
   - **Without templates:** More iterations (15-30) may be needed

4. **Handle noise appropriately:**
   - Use `denoise: true` if data appears noisy from visualization
   - Lower `maxsize` to avoid overfitting to noise
   - Templates help avoid overfitting by constraining the hypothesis space

5. **For multi-variable systems:**
   - Think about variable groupings: which variables interact? which are independent?
   - If variables have separable effects based on domain knowledge, use templates to decompose
   - DO NOT mechanically split variables - only separate when physically/logically justified
