# Benchmark Evaluation Guide

This guide explains how to evaluate the AgentSR system on the LLM-SRBench datasets.

## Quick Start

```bash
# Navigate to agentsr directory
cd agentsr

# Make script executable (first time only)
chmod +x evaluate_benchmark.sh

# Run evaluation on a category
./evaluate_benchmark.sh bio_pop_growth
```

## Available Categories

The LLM-SRBench dataset includes five categories:

1. **bio_pop_growth** - Biological population growth equations (~15 datasets)
2. **chem_react** - Chemical reaction equations (~25 datasets)
3. **matsci** - Materials science equations (~17 datasets)
4. **phys_osc** - Physics oscillation equations (~30 datasets)
5. **lsr_transform** - Transformed Feynman equations (~110 datasets, large set)

## Usage

### Basic Usage

```bash
# Evaluate on a single category
./evaluate_benchmark.sh <category>

# Examples:
./evaluate_benchmark.sh bio_pop_growth
./evaluate_benchmark.sh lsr_transform
```

### Advanced Options

```bash
./evaluate_benchmark.sh <category> [options]

Options:
  --skip-symbolic      Skip symbolic accuracy evaluation (faster)
  --eval-model MODEL   Model for symbolic evaluation (default: gpt-4o)
  --agent-model MODEL  Model for SR agent (default: gpt-4o-mini)
  --temperature TEMP   Temperature for agent LLM (default: 0.7)
  --max-iter N         Maximum iterations (default: 5)
  --parallel N         Number of parallel jobs (1-12, default: 1)
  --resume             Resume from existing results (skip completed datasets)
  --limit N            Limit to first N datasets (for testing)
  --help               Show this help message
```

### Common Use Cases

#### 1. Test Run (Quick)
```bash
# Evaluate first 5 datasets, skip symbolic check
./evaluate_benchmark.sh bio_pop_growth --skip-symbolic --limit 5
```

#### 2. Full Evaluation with Symbolic Checking
```bash
# Complete evaluation including symbolic accuracy
./evaluate_benchmark.sh bio_pop_growth
```

#### 3. Fast Numerical-Only Evaluation
```bash
# Skip symbolic checking for speed (numerical metrics only)
./evaluate_benchmark.sh lsr_transform --skip-symbolic
```

#### 4. Resume Interrupted Evaluation
```bash
# Skip already completed datasets
./evaluate_benchmark.sh phys_osc --resume
```

#### 5. Use Better Models
```bash
# Use GPT-4o for both agent and evaluation
./evaluate_benchmark.sh chem_react \
  --agent-model gpt-4o \
  --eval-model gpt-4o
```

#### 6. Large-Scale Batch Processing with Parallel Execution
```bash
# For the large lsr_transform category, use faster settings with parallelization
./evaluate_benchmark.sh lsr_transform \
  --parallel 12 \
  --skip-symbolic \
  --agent-model gpt-4o-mini \
  --max-iter 3
```

#### 7. Maximum Speed Evaluation
```bash
# Use all optimizations: 12 parallel jobs, skip symbolic, minimal iterations
./evaluate_benchmark.sh lsr_transform \
  --parallel 12 \
  --skip-symbolic \
  --max-iter 3
```

## Output Structure

Results are saved to `agentsr/evaluation_results/<category>/`:

```
agentsr/evaluation_results/
├── bio_pop_growth/
│   ├── BPG0_eval.json          # Individual result files
│   ├── BPG1_eval.json
│   ├── ...
│   ├── evaluation_log_<timestamp>.txt   # Detailed log
│   └── evaluation_summary.txt           # Summary statistics
├── chem_react/
├── matsci/
├── phys_osc/
└── lsr_transform/
```

### Individual Result File Format

Each `<dataset>_eval.json` contains:

```json
{
  "discovered_equation": "x**2 + 2*y",
  "ground_truth_equation": "a*x**2 + b*y",
  "symbolic_accuracy": {
    "is_equivalent": true,
    "reasoning": "..."
  },
  "numerical_metrics": {
    "test": {
      "success": true,
      "metrics": {
        "mse": 1.23e-05,
        "nmse": 2.34e-06,
        "r2": 0.999999,
        "kdt": 0.999,
        "mape": 0.001
      }
    }
  },
  "summary": {...}
}
```

### Evaluation Log

The log file (`evaluation_log_<timestamp>.txt`) contains:
- Complete console output
- Progress for each dataset
- Success/failure status
- Any error messages

### Evaluation Summary

The summary file (`evaluation_summary.txt`) contains:
- Category and configuration
- Success rate
- Total/successful/failed counts
- Timestamp and paths

## Example Workflows

### 1. Complete Benchmark Evaluation

Evaluate all categories with full symbolic + numerical metrics:

```bash
#!/bin/bash
for category in bio_pop_growth chem_react matsci phys_osc lsr_transform; do
  echo "Evaluating $category..."
  ./evaluate_benchmark.sh "$category"
done
```

### 2. Quick Baseline (Numerical Only) with Parallelization

Fast evaluation of all categories without symbolic checking, using parallel execution:

```bash
#!/bin/bash
for category in bio_pop_growth chem_react matsci phys_osc lsr_transform; do
  echo "Evaluating $category (numerical only)..."
  ./evaluate_benchmark.sh "$category" --parallel 12 --skip-symbolic
done
```

### 3. Test on Small Subsets

Test on first 3 datasets of each category:

```bash
#!/bin/bash
for category in bio_pop_growth chem_react matsci phys_osc; do
  echo "Testing $category..."
  ./evaluate_benchmark.sh "$category" --limit 3 --skip-symbolic
done
```

### 4. High-Quality Evaluation with Parallelization

Use best models for comprehensive evaluation, with parallel execution:

```bash
./evaluate_benchmark.sh lsr_transform \
  --parallel 8 \
  --agent-model gpt-4o \
  --eval-model gpt-4o \
  --temperature 0.3 \
  --max-iter 10
```

**Note**: When using gpt-4o with symbolic evaluation, consider lower parallelization (4-8 jobs) to manage API rate limits and costs.

## Performance Considerations

### Speed Optimization

The script now supports parallel execution with up to **12 concurrent jobs**!

- **Parallel execution**: Process multiple datasets simultaneously (up to 12x speedup)
- **Skip symbolic**: Saves ~1-2 seconds per dataset
- **Lower max-iter**: Fewer tool iterations = faster completion
- **Smaller model**: gpt-4o-mini is faster than gpt-4o

Example fast configuration with parallelization:
```bash
./evaluate_benchmark.sh lsr_transform \
  --parallel 12 \
  --skip-symbolic \
  --agent-model gpt-4o-mini \
  --max-iter 3
```

### Parallel Execution Details

The script automatically handles parallel execution using either:
1. **GNU parallel** (preferred, more efficient)
2. **xargs -P** (fallback if GNU parallel not available)

Each parallel job:
- Runs independently in its own process
- Has its own log file (later aggregated)
- Reports progress and results to a shared results file

**Recommended parallelization levels:**
- Small categories (bio_pop_growth, matsci): `--parallel 4-6`
- Medium categories (chem_react, phys_osc): `--parallel 8-10`
- Large category (lsr_transform): `--parallel 12`

### Cost Optimization

To minimize API costs:
1. Use `gpt-4o-mini` for agent
2. Skip symbolic evaluation (`--skip-symbolic`)
3. Limit iterations (`--max-iter 3`)
4. Test on small subset first (`--limit 5`)

### Accuracy Optimization

For best results:
1. Use `gpt-4o` for agent
2. Enable symbolic evaluation (default)
3. Increase iterations (`--max-iter 10`)
4. Lower temperature (`--temperature 0.3`)

## Monitoring Progress

### Real-time Monitoring

With parallel execution, multiple datasets are evaluated simultaneously. The script shows progress from each parallel job:

```
[INFO] Processing 15 datasets with 4 parallel jobs...
[INFO] Using GNU parallel for execution
...
[SUCCESS] [5/15] Completed: BPG5 at Wed Jan 06 12:30:45 2026
[SUCCESS] [3/15] Completed: BPG3 at Wed Jan 06 12:30:46 2026
...
[INFO] All parallel jobs completed
```

### Check Results During Execution

While the script is running, you can check partial results:

```bash
# Count completed evaluations
ls -1 agentsr/evaluation_results/bio_pop_growth/*_eval.json | wc -l

# View individual dataset logs (during parallel execution)
tail -f agentsr/evaluation_results/bio_pop_growth/BPG*_log.txt

# View aggregated log (after completion)
tail -f agentsr/evaluation_results/bio_pop_growth/evaluation_log_*.txt

# Check success rate in real-time
grep -c "SUCCESS:" agentsr/evaluation_results/bio_pop_growth/.evaluation_results_*.txt

# Check for failures
grep "FAILED:" agentsr/evaluation_results/bio_pop_growth/.evaluation_results_*.txt
```

## Analyzing Results

### Aggregate Statistics

After evaluation, analyze results with Python:

```python
import json
import glob
from pathlib import Path

# Load all results for a category
category = "bio_pop_growth"
result_files = glob.glob(f"evaluation_results/{category}/*_eval.json")

# Compute aggregate metrics
total = 0
symbolic_correct = 0
r2_scores = []

for file in result_files:
    with open(file) as f:
        result = json.load(f)
        total += 1

        # Symbolic accuracy
        if result.get("symbolic_accuracy"):
            if result["symbolic_accuracy"]["is_equivalent"]:
                symbolic_correct += 1

        # R² score
        if "test" in result["numerical_metrics"]:
            if result["numerical_metrics"]["test"]["success"]:
                r2 = result["numerical_metrics"]["test"]["metrics"]["r2"]
                r2_scores.append(r2)

print(f"Category: {category}")
print(f"Total: {total}")
print(f"Symbolic Accuracy: {symbolic_correct}/{total} ({symbolic_correct/total*100:.1f}%)")
print(f"Average R²: {sum(r2_scores)/len(r2_scores):.4f}")
```

### Compare Across Categories

```python
import json
import glob
import pandas as pd

categories = ["bio_pop_growth", "chem_react", "matsci", "phys_osc"]
results = []

for category in categories:
    files = glob.glob(f"evaluation_results/{category}/*_eval.json")

    for file in files:
        with open(file) as f:
            data = json.load(f)

            row = {
                "category": category,
                "dataset": Path(file).stem.replace("_eval", ""),
                "symbolic_match": data.get("symbolic_accuracy", {}).get("is_equivalent", False),
            }

            if "test" in data["numerical_metrics"]:
                metrics = data["numerical_metrics"]["test"]["metrics"]
                row.update({
                    "r2": metrics["r2"],
                    "mse": metrics["mse"],
                    "nmse": metrics["nmse"]
                })

            results.append(row)

df = pd.DataFrame(results)
print(df.groupby("category").mean())
```

## Troubleshooting

### Script Fails Immediately

**Issue**: Permission denied
```bash
chmod +x evaluate_benchmark.sh
```

**Issue**: Data directory not found
- Check that datasets are extracted: `cd llmsr/data/llmsrbench/csv`
- Run dataset extraction if needed

### Evaluation Fails for Some Datasets

**Issue**: API rate limits
- Add delays between datasets (modify script)
- Use `--resume` to continue from where it stopped

**Issue**: Out of memory
- Reduce `--max-iter`
- Close other applications

### Results Missing

**Issue**: No evaluation results saved
- Check if `--eval` flag is properly passed to main.py (it is, this is built into the script)
- Check permissions on output directory
- Review log file for errors

## Advanced Usage

### Built-in Parallel Execution

The script now has **built-in parallelization** within each category:

```bash
# Evaluate a category with 12 parallel jobs
./evaluate_benchmark.sh lsr_transform --parallel 12 --skip-symbolic
```

### Multi-Category Parallel Execution

For even faster processing, run multiple categories simultaneously:

```bash
# Run different categories in separate terminals or use screen/tmux
./evaluate_benchmark.sh bio_pop_growth --parallel 8 &
./evaluate_benchmark.sh chem_react --parallel 8 &
./evaluate_benchmark.sh matsci --parallel 8 &
wait
```

**Warning**: Be mindful of API rate limits when running too many parallel jobs across multiple categories!

### Custom Evaluation Script

For more control, create a custom script:

```bash
#!/bin/bash

CATEGORY="bio_pop_growth"
OUTPUT_DIR="evaluation_results/$CATEGORY"
mkdir -p "$OUTPUT_DIR"

cd agentsr/src

for dataset in BPG0 BPG1 BPG2; do
  python main.py -D "$dataset" \
    --eval \
    --model gpt-4o-mini \
    --output-dir "$OUTPUT_DIR"
done
```

## Environment Variables

Set these for better control:

```bash
# OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Logging level
export AGENTSR_LOG_LEVEL="INFO"  # or DEBUG for verbose output
```

## Best Practices

1. **Start small**: Test with `--limit 3` first
2. **Use resume**: Always use `--resume` for large evaluations
3. **Monitor costs**: Check API usage regularly
4. **Save logs**: Keep log files for debugging
5. **Version control**: Track configuration in summary files

## Help and Support

For help:
```bash
./evaluate_benchmark.sh --help
```

For issues, check:
- Log files in `evaluation_results/<category>/`
- Main script documentation: [main.py](src/main.py)
- Evaluation module docs: [src/evaluation/README.md](src/evaluation/README.md)
