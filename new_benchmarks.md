## Adding and Running New Benchmarks

1. **Create a dataset folder** under `data/` (e.g., `data/my_benchmark/`) containing `train.csv`, `test_id.csv`, and `test_ood.csv`. Use the format `feature_0,...,feature_n,target` with the last column as the output label.
2. **Author a specification** by copying one of the templates in `specs/` (e.g., `specs/specification_oscillator1_numpy.txt`) and tailoring the docstring, variable names, parameter count, and starter skeleton to the new scientific context.
3. **Define the evaluation + equation hooks** just like `specs/strogatz_numpy.txt`: expose `@evaluate.run` that ingests a `(inputs, outputs)` dict, optimizes parameters (SciPy’s `minimize` is used in the template), and returns a scalar score; expose `@equation.evolve` that accepts two feature columns (`x1`, `x2`) plus a parameter vector and computes the model output. Keep `MAX_NPARAMS`, `PRAMS_INIT`, and any optimizer defaults aligned with the data you prepared in step 1.


Once the folder and spec exist, you can run the benchmark with either a local LLM or an API-backed one. Replace the placeholder names below with your own paths:

```
# Local / self-hosted LLM
python main.py \
    --problem_name data/my_benchmark \
    --spec_path ./specs/specification_my_benchmark_numpy.txt \
    --log_path ./logs/my_benchmark_local

# API-backed run (e.g., OpenAI)
export API_KEY=sk-...
python main.py \
    --use_api True \
    --api_model "gpt-4.1-mini" \
    --problem_name data/my_benchmark \
    --spec_path ./specs/specification_my_benchmark_numpy.txt \
    --log_path ./logs/my_benchmark_api
```

Tips:
- Keep `problem_name` pointed at the dataset folder (the defaults live in `data/`).
- Update `sampler.py` if you host the LLM on a non-default port.
- Use `logs/` to track multiple runs; the directory is created automatically.
- For reproducibility, record any stochastic data-generation seeds in your new benchmark folder or spec docstring.

## Dummy Data Node

When your raw measurements arrive as `.txt` dumps, you can turn them into an LLMSR-ready benchmark with:

```
python scripts/dummy_data_node.py \
    --input-dir raw_data \
    --output-root data \
    --problem-name my_benchmark \
    --train-frac 0.7 \
    --test-id-frac 0.15 \
    --normalize  # optional min–max scaling
```

The script concatenates every `.txt` file in `raw_data/`, renames the columns to `feature_i`/`target`, optionally normalizes all columns to `[0, 1]`, and writes `train.csv`, `test_id.csv`, and `test_ood.csv` under `data/my_benchmark/`.


