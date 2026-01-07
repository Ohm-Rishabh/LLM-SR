# Symbolic Regression Evaluation Module

A comprehensive evaluation pipeline for symbolic regression results, providing both symbolic accuracy and numerical precision metrics.

## Overview

This module implements three types of evaluation:

1. **Symbolic Accuracy**: Uses GPT-4o to determine if the discovered equation is symbolically equivalent to the ground truth (following LLM-SRBench approach)
2. **Numerical Precision**: Computes standard regression metrics (MSE, NMSE, R², Kendall Tau, MAPE)
3. **Unified Evaluation**: Combines both symbolic and numerical evaluation in a single interface

## Installation

The evaluation module requires the following dependencies:

```bash
pip install numpy scipy scikit-learn sympy openai
```

Make sure you have your OpenAI API key set:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Basic Usage

```python
from evaluation import SRAgentEvaluator
import numpy as np

# Initialize evaluator
evaluator = SRAgentEvaluator()

# Prepare test data (format: [output, input1, input2, ...])
test_data = np.column_stack([y_true, x1, x2])

# Evaluate
results = evaluator.evaluate(
    discovered_equation="x1**2 + 2*x2",
    ground_truth_equation="a*x1**2 + b*x2",
    test_data=test_data,
    symbols=["y", "x1", "x2"],
    check_symbolic=True
)

# Access results
print(f"Symbolically equivalent: {results['symbolic_accuracy']['is_equivalent']}")
print(f"Test R²: {results['numerical_metrics']['test']['metrics']['r2']}")
```

### With Agent Output

```python
from evaluation import SRAgentEvaluator

evaluator = SRAgentEvaluator()

# Evaluate directly from agent output and dataset metadata
results = evaluator.evaluate_from_agent_output(
    agent_output={"final_result": "x**2 + y"},
    dataset_metadata=metadata,
    test_data=test_data,
    check_symbolic=True
)

# Save results
evaluator.save_results(results, "evaluation_results.json")
```

### Skip Symbolic Evaluation

For faster evaluation, you can skip symbolic checking:

```python
results = evaluator.evaluate(
    discovered_equation="x**2 + y",
    ground_truth_equation="x**2 + y",
    test_data=test_data,
    symbols=["z", "x", "y"],
    check_symbolic=False  # Skip symbolic evaluation
)
```

## Module Components

### 1. SymbolicAccuracyEvaluator

Evaluates symbolic equivalence using GPT-4o as an LLM judge.

```python
from evaluation.symbolic_evaluator import SymbolicAccuracyEvaluator

evaluator = SymbolicAccuracyEvaluator(
    model="gpt-4o",
    temperature=0.0
)

result = evaluator.evaluate(
    ground_truth="a*x**2 + b*y",
    hypothesis="3*x**2 + 2*y"
)

print(result['is_equivalent'])  # True/False
print(result['reasoning'])       # Explanation from GPT-4o
```

**Prompt Template:**

The evaluator uses the following prompt (from LLM-SRBench):

> Given the ground truth mathematical expression A and the hypothesis B, determine if there exist any constant parameter values that would make the hypothesis equivalent to the given ground truth expression.
>
> Let's think step by step. Explain your reasoning and then provide the final answer as:
> ```json
> {
>   "reasoning": "brief step-by-step analysis",
>   "answer": "yes/no"
> }
> ```

### 2. NumericalEvaluator

Computes numerical metrics by evaluating equations on data.

```python
from evaluation.numerical_evaluator import NumericalEvaluator
import numpy as np

evaluator = NumericalEvaluator()

# Evaluate equation on data
result = evaluator.evaluate_equation_string(
    equation_str="x**2 + 2*y",
    X=np.array([[1, 2], [3, 4]]),  # Input features
    y_true=np.array([5, 17]),       # True outputs
    symbols=["x", "y"]
)

metrics = result['metrics']
print(f"MSE: {metrics['mse']}")
print(f"R²: {metrics['r2']}")
```

**Metrics Computed:**

- **MSE** (Mean Squared Error): Average squared difference
- **NMSE** (Normalized MSE): MSE divided by variance
- **R²** (Coefficient of Determination): Proportion of variance explained
- **KDT** (Kendall Tau): Rank correlation coefficient
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
- **Accuracy to Tolerance**: Binary metric (1 if max relative error ≤ 0.1, else 0)
- **Max Relative Error**: Maximum relative error across all test points

### 3. SRAgentEvaluator

Unified interface combining both symbolic and numerical evaluation.

```python
from evaluation import SRAgentEvaluator

evaluator = SRAgentEvaluator(
    symbolic_model="gpt-4o",
    symbolic_temperature=0.0
)

results = evaluator.evaluate(
    discovered_equation="x**2 + y",
    ground_truth_equation="a*x**2 + b*y",
    test_data=test_data,
    symbols=["z", "x", "y"],
    train_data=train_data,      # Optional
    ood_test_data=ood_data,     # Optional
    check_symbolic=True
)
```

**Result Structure:**

```json
{
  "discovered_equation": "x**2 + 2*y",
  "ground_truth_equation": "a*x**2 + b*y",
  "symbolic_accuracy": {
    "is_equivalent": true,
    "reasoning": "...",
    "raw_response": "..."
  },
  "numerical_metrics": {
    "test": {
      "success": true,
      "metrics": {
        "mse": 0.0,
        "nmse": 0.0,
        "r2": 1.0,
        "kdt": 1.0,
        "mape": 0.0,
        "num_valid_points": 100
        "accuracy_to_tolerance": 1.0,
        "max_relative_error": 0.0,
      }
    }
  },
  "summary": {
    "is_symbolically_equivalent": true,
    "test_metrics": {
      "mse": 0.0,
      "nmse": 0.0,
      "r2": 1.0,
      "num_valid_points": 100
        "accuracy_to_tolerance": 1.0,
        "max_relative_error": 0.0,
    }
  }
}
```

## Integration with Agent Pipeline

### Using main.py with --eval flag

The main script supports optional evaluation via the `--eval` flag:

```bash
# Run with evaluation
python main.py -D I.10.7_1_0 --eval

# Skip symbolic evaluation (faster)
python main.py -D I.10.7_1_0 --eval --skip-symbolic

# Specify output directory
python main.py -D I.10.7_1_0 --eval --output-dir ./results

# Use different models
python main.py -D I.10.7_1_0 --eval \
    --model gpt-4o \
    --eval-model gpt-4o

# Run without evaluation (default)
python main.py -D I.10.7_1_0
```

### Custom Integration in Your Code

```python
from evaluation import SRAgentEvaluator
from datasets.llmsrbench import LLMSRBenchDataset
import pandas as pd

# Load dataset
dataset_manager = LLMSRBenchDataset()
train_csv, metadata = dataset_manager.get_dataset("I.10.7_1_0", "train")
test_csv, _ = dataset_manager.get_dataset("I.10.7_1_0", "test")

# Load test data
test_df = pd.read_csv(test_csv)
test_data = test_df.values

# ... run your agent workflow ...

# Evaluate results
evaluator = SRAgentEvaluator()
results = evaluator.evaluate_from_agent_output(
    agent_output=workflow_result,
    dataset_metadata=metadata,
    test_data=test_data
)

# Save results
evaluator.save_results(results, "evaluation.json")
```

## Testing

Run the test suite to verify installation:

```bash
cd agentsr/src/evaluation
python test_evaluator.py
```

This will run 5 test cases:
1. Perfect match between discovered and ground truth
2. Equivalent equations with different constants
3. Approximate match (small numerical error)
4. Completely wrong equation
5. Numerical-only evaluation

## Advanced Usage

### Custom Symbolic Evaluation Model

```python
evaluator = SRAgentEvaluator(
    symbolic_model="gpt-4o-mini",  # Use smaller model
    symbolic_temperature=0.2       # Add some randomness
)
```

### Handling Multiple Data Splits

```python
results = evaluator.evaluate(
    discovered_equation="x**2 + y",
    ground_truth_equation="x**2 + y",
    test_data=test_data,
    train_data=train_data,         # Also evaluate on train
    ood_test_data=ood_test_data,   # Also evaluate on OOD
    symbols=["z", "x", "y"]
)

# Access different splits
train_r2 = results['numerical_metrics']['train']['metrics']['r2']
test_r2 = results['numerical_metrics']['test']['metrics']['r2']
ood_r2 = results['numerical_metrics']['ood_test']['metrics']['r2']
```

### Error Handling

The evaluators are designed to handle errors gracefully:

```python
result = evaluator.evaluate_equation_string(
    equation_str="invalid syntax!",
    X=X, y_true=y, symbols=["x"]
)

if not result['success']:
    print(f"Evaluation failed: {result['error']}")
```

## Data Format

### Test Data Format

Test data should be a numpy array of shape `(n_samples, n_features + 1)`:
- **First column**: Output variable (y)
- **Remaining columns**: Input features (x1, x2, ...)

```python
# Example: y = x1^2 + x2
test_data = np.column_stack([
    y,   # Output
    x1,  # Input 1
    x2   # Input 2
])
```

### Symbols Format

Symbols list should match the data columns:

```python
symbols = ["y", "x1", "x2"]  # Output first, then inputs
```

## Performance Considerations

- **Symbolic evaluation** makes an API call to GPT-4o (~1-2 seconds per evaluation)
- **Numerical evaluation** is fast (~milliseconds)
- For batch evaluation, consider:
  - Using `check_symbolic=False` for initial filtering
  - Running symbolic evaluation only on top candidates
  - Using `gpt-4o-mini` for faster (but less accurate) symbolic checking

## References

This implementation follows the approach from:

- **LLM-SRBench**: Symbolic equivalence checking with LLM evaluators
- **Standard SR Metrics**: MSE, NMSE, R², Kendall Tau from scikit-learn and scipy

## Additional Documentation

- **[Accuracy to Tolerance Metric](ACCURACY_TO_TOLERANCE.md)**: Detailed explanation of the LLM-SRBench accuracy metric

## License

Part of the LLM-SR project.
