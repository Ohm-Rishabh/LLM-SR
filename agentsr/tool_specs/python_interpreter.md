#### Description

**python_interpreter** evaluates Python code and returns the results. This tool is intended specifically for data analysis tasks. This tool is usually the choice in the first step to obtain basic insights of the dataset, but may also be called in later stages if additional data analysis is necessary.

**Use Cases:**
- Exploratory data analysis (statistics, distributions, correlations)
- Data visualization (plots, charts, graphs)
- Computing descriptive statistics and metrics
- Preliminary analysis to inform symbolic regression strategies

#### Usage Guidelines
- This tool is for **data analysis only**, not for symbolic regression itself
- The agent should write their own code for loading data files and exploring data patterns, not implementing SR algorithms
- Output files (plots, results) should be saved to the workspace output directory. Only save unreadable results (e.g., denoised data file in binary format, image) to files.
- The code should **print** the analysis results to stdout using `print()` statements. The output will be captured and returned as the tool result.
- You can use multiple print statements to output different pieces of information.
- The printed output should be human-readable and contain all insights obtained from the analysis.

**Output Format Guidelines:**
- Print results in a clear, readable format (formatted strings, tables, statistics, etc.)
- Include all relevant insights: statistics, correlations, observations, etc.
- If you saved any files (plots, data files, etc.), mention the filenames and their descriptions in the output
- Avoid printing raw complex objects (DataFrames, arrays) - instead describe their key characteristics

**Example output format:**
```
Dataset shape: 1000 rows, 5 columns
Missing values: None

Summary statistics:
- Variable A: mean=10.5, std=2.3
- Variable B: mean=20.1, std=4.5

Correlation between A and B: 0.85 (strong positive)

Saved files:
- correlation_matrix.png: Heatmap of all variable correlations
- distribution_plots.png: Histograms for all variables
```

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
data_file = os.path.join(workspace_input, "data.csv")
df = pd.read_csv(data_file)

# Perform analysis
corr = df['A'].corr(df['B'])
mean_a = df['A'].mean()
mean_b = df['B'].mean()

# Print analysis results
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print()
print("Summary statistics:")
print(f"- Variable A: mean={mean_a:.2f}, std={df['A'].std():.2f}")
print(f"- Variable B: mean={mean_b:.2f}, std={df['B'].std():.2f}")
print()
print(f"Correlation between A and B: {corr:.3f}")

# If you save files, mention them
# plt.savefig(os.path.join(workspace_output, "plot.png"))
# print("\nSaved files:")
# print("- plot.png: Scatter plot of A vs B")
```

**Important**: Unlike other tools, you must not wrap the Python code in the JSON object as one of the arguments. You should create a separate Python code block in your response as above. Make sure you provide **both the JSON object specifying this tool and the Python code**.
