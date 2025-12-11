#### Description

**python_interpreter** evaluates Python code and returns the results. This tool is intended specifically for data analysis tasks. This tool is usually the choice in the first step to obtain basic insights of the dataset, but may also be called in later stages if additional data analysis is necessary.

**Use Cases:**
- Exploratory data analysis (statistics, distributions, correlations)
- Data visualization (plots, charts, graphs)
- Computing descriptive statistics and metrics
- Data quality checks and validation
- Feature engineering exploration
- Preliminary analysis to inform symbolic regression strategies

#### Usage Guidelines
- This tool is for **data analysis only**, not for symbolic regression itself
- The agent should write their own code for loading data files and exploring data patterns, not implementing SR algorithms
- Output files (plots, results) should be saved to the workspace output directory. Only save unreadable results (e.g., denoised data file in binary format, image) to files.
- The code should save the result in the `result` variable, which should be a JSON object with the following format:
```json
{
  "summary": "All insights obtained from the code, e.g. the means of variables A and B are ...; the correlation of variable C and D are ...",
  "saved_files": // all files saved in the executed code, if any
  {
    "filename1": "file description 1",
    "filename2": "file description 2",
    // ...
  }
}
```

**IMPORTANT**: The `summary` field must contain **only JSON-serializable, human-readable values** (strings, numbers, booleans, lists, dicts). **Do NOT include** pandas DataFrames, NumPy arrays, or other complex objects directly in the summary. Instead:
- Extract scalar values from NumPy arrays (e.g., `float(array.mean())`)
- Describe results in plain text with key statistics
- Save complex objects to files and reference the file paths in `saved_files`

**Pre-imported Libraries and Environment Variables**

The following libraries are pre-imported and ready to use:
- `numpy` (as `np`)
- `pandas` (as `pd`)
- `matplotlib.pyplot` (as `plt`)
- `seaborn` (as `sns`)
- `scipy.stats` (as `stats`)
- `os`

The following variables are already defined, storing the paths to the workspace:
- `workspace_root`: Main workspace directory
- `workspace_input`: Input files directory (where data files are stored)
- `workspace_output`: Output directory for saving plots and results
- `workspace_logs`: Log files directory
- `workspace_scratch`: Temporary/scratch directory

#### Example Usage
```json
{
  "tool_call":
  {
    "tool_name": "python_interpreter",
  }
}
```

```python
# Load data from workspace
data_file = os.path.join(os.environ['WORKSPACE_INPUT'], "data.csv")
df = pd.read_csv(data_file)

# Perform analysis
# ...

# write results
summary = [
  "The dataset is clean without any missing values",
  f"The correlation between variables A and B is {corr}",
  # ...
]
result = {
  "summary": summary,
  "saved_files": {
    # ...
  }
}
```

**Important**: Unlike other tools, you must not wrap the Python code in the JSON object as one of the arguments. You should create a separate Python code block in your response as above. Make sure you provide **both the JSON object specifying this tool and the Python code**.
