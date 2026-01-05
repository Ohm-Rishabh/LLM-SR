# PySR Tool Tests

This directory contains unit tests for the PySR tool.

## Running Tests

To run all tests:

```bash
cd /home/ubuntu/LLM-SR/agentsr/src/tools/pysr/tests
python test_template_combine.py
```

Or run with verbose output:

```bash
python test_template_combine.py -v
```

## Test Modules

### `test_template_combine.py`

Tests the `combine_template_equation` function from `template_utils.py`, which combines PySR's template sub-expressions into human-readable final equations.

**Test Coverage:**
- Simple additive templates
- Nested trigonometric templates
- Exponential templates
- Multiplicative templates
- Complex physics-inspired templates
- Handling of newline-separated equations
- Preservation of constants in sub-expressions
- Edge cases (no template, single variables, etc.)

## Adding New Tests

To add new test cases:

1. Create a new test method in the appropriate `TestCase` class
2. Follow the naming convention: `test_<descriptive_name>`
3. Use descriptive docstrings
4. Include assertions with clear expected values

Example:

```python
def test_my_new_case(self):
    """Test description here"""
    equation = "f = #1 + #2"
    expression_spec = {
        "expressions": ["f"],
        "variable_names": ["x", "y"],
        "combine": "f(x, y)"
    }
    feature_names = ["x", "y"]
    expected = "(x + y)"

    result = combine_template_equation(equation, expression_spec, feature_names)
    self.assertEqual(result, expected)
```
