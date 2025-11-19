## Adding and Running New Benchmarks

1. **Create a dataset folder** under `data/` (e.g., `data/my_benchmark/`) containing `train.csv`, `test_id.csv`, and `test_ood.csv`. Use the format `feature_0,...,feature_n,target` with the last column as the output label.
2. **Author a specification** by copying one of the templates in `specs/` (e.g., `specs/specification_oscillator1_numpy.txt`) and tailoring the docstring, variable names, parameter count, and starter skeleton to the new scientific context.
3. **Define the evaluation + equation hooks** just like `specs/strogatz_numpy.txt`: expose `@evaluate.run` that ingests a `(inputs, outputs)` dict, optimizes parameters (SciPy’s `minimize` is used in the template), and returns a scalar score; expose `@equation.evolve` that accepts two feature columns (`x1`, `x2`) plus a parameter vector and computes the model output. Keep `MAX_NPARAMS`, `PRAMS_INIT`, and any optimizer defaults aligned with the data you prepared in step 1.


Once the folder and spec exist, you can run the benchmark with either a local LLM or an API-backed one. Replace the placeholder names below with your own paths:
## Local Runs (Open-Source LLMs)

### Start the local LLM Server

First, start the local LLM engine from huggingface models by using the `bash run_server.sh` script or running the following command: 

```
cd llm_engine

python engine.py --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
                    --gpu_ids [GPU_ID]  \
                    --port [PORT_ID] --quantization
```

* Set `gpu_ids` and `port` parameters based on your server availability

* Change `model_path` to use a different open-source model from Hugging Face

* `quantization` activates efficient inference of LLM with quantization on GPUs

* Control quantization level with `load_in_4bit` and `load_in_8bit` parameters in [engine.py](./llm_engine/engine.py)



### Run LLM-SR on Local Server
After activating the local LLM server, run the LLM-SR framework on your dataset with the `run_llmsr.sh` script or running the following command: 

```
python main.py --problem_name [PROBLEM_NAME] \
                   --spec_path [SPEC_PATH] \
                   --log_path [LOG_PATH]
```

* Update the `port` id in the url in [sampler.py](./llmsr/sampler.py) to match the LLM server port

* `problem_name` refers to the target problem and dataset in [data/](./data)

* `spec_path` refers to the initial prompt specification file path in [spec/](./specs) 

* Available problem names for datasets: `oscillator1`, `oscillator2`, `bactgrow`, `stressstrain`

For more example scripts, check `run_llmsr.sh`. 



## API Runs (Closed LLMs)
To run LLM-SR with the OpenAI GPT API, use the following command: 

```
export API_KEY=[YOUR_API_KEY_HERE]

python main.py --use_api True \
                   --api_model "gpt-3.5-turbo" \
                   --problem_name [PROBLEM_NAME] \
                   --spec_path [SPEC_PATH] \
                   --log_path [LOG_PATH]
```

* Replace `[YOUR_API_KEY_HERE]` with your actual OpenAI API key to set your API key as an environment variable. 

* `--use_api True`: Enables the use of the OpenAI API instead of local LLMs from Hugging Face

* `--api_model`: Specifies the GPT model to use (e.g., "gpt-3.5-turbo", "gpt-4o")

* `--problem_name`, `--spec_path`, `--log_path`: Set these as in the local runs section


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


