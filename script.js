// Global variable to store the CSV data
let csvData;
let pyodide;
let derivedMetrics = []; // Store derived metrics

// Define loading template
const createLoadingSpinner = (message = "Loading...") => {
    const spinner = document.createElement('div');
    spinner.className = 'text-center my-3';
    spinner.innerHTML = `
        <div class="spinner-border" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">${message}</p>
    `;
    return spinner;
};

// Function to preview CSV data
async function previewCSV() {
    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please upload a CSV file.");
        return;
    }

    // Show loading spinner in the preview section
    const tablePreview = document.getElementById("tablePreview");
    tablePreview.innerHTML = '';
    tablePreview.appendChild(createLoadingSpinner("Loading data preview..."));
    document.getElementById("previewSection").classList.remove("hidden");

    // Initialize Pyodide if not already done
    if (!pyodide) {
        document.getElementById("previewButton").innerText = "Loading Pyodide...";
        pyodide = await loadPyodide();
        await pyodide.loadPackage("pandas");
        document.getElementById("previewButton").innerText = "Preview Data";
    }

    let reader = new FileReader();
    reader.onload = async function (event) {
        csvData = event.target.result;

        // Use Python to read and display the CSV preview
        let script = `
import pandas as pd
from io import StringIO
import json

csv_data = """${csvData}"""
df = pd.read_csv(StringIO(csv_data))

# Get column names for dropdown
columns = df.columns.tolist()

# Get data types and sample values for schema
schema = []
for col in df.columns:
    sample_values = df[col].dropna().head(3).tolist()
    sample_values = [str(val) for val in sample_values]
    schema.append({
        "column": col,
        "dtype": str(df[col].dtype),
        "sample_values": sample_values
    })

# Get preview HTML
preview_html = df.head().to_html()

# Return both as a JSON string
result_dict = {"preview": preview_html, "columns": columns, "schema": schema}
json.dumps(result_dict)
        `;

        let resultStr = await pyodide.runPythonAsync(script);
        let result = JSON.parse(resultStr);

        // Display the table preview
        tablePreview.innerHTML = result.preview;

        // Populate the dropdown with column names
        const targetDropdown = document.getElementById("targetColumn");
        targetDropdown.innerHTML = "";
        result.columns.forEach(column => {
            const option = document.createElement("option");
            option.value = column;
            option.textContent = column;
            targetDropdown.appendChild(option);
        });

        // Store the schema for LLM processing
        window.datasetSchema = result.schema;
        
        // Show the derived metrics button
        document.getElementById("derivedMetricsSection").classList.remove("hidden");
    };

    reader.readAsText(fileInput.files[0]);
}

// Function to get derived metrics suggestions from LLM
async function getDerivedMetricsSuggestions() {
    if (!window.datasetSchema) {
        alert("Please upload and preview a CSV file first.");
        return;
    }

    // Show loading spinner
    const metricsContainer = document.getElementById("derivedMetricsContainer");
    metricsContainer.innerHTML = '';
    metricsContainer.appendChild(createLoadingSpinner("Getting suggestions from AI..."));
    
    try {
        // Get token from LLM Foundry
        const { token } = await fetch("https://llmfoundry.straive.com/token", {
            credentials: "include",
        }).then((res) => res.json());
        
        if (!token) {
            const url = "https://llmfoundry.straive.com/login?" + new URLSearchParams({ next: location.href });
            metricsContainer.innerHTML = /* html */ `<div class="text-center my-5"><a class="btn btn-lg btn-primary" href="${url}">Log in to analyze</a></div>`;
            throw new Error("User is not logged in");
        }
        
        // Prepare the prompt for the LLM
        const systemPrompt = "You are a data analysis expert. Based on the dataset schema provided, suggest useful derived metrics or features that could improve analysis. For each suggestion, provide a name, description, and the Python code to calculate it.";
        
        const userPrompt = `Here is the schema of my dataset:
${JSON.stringify(window.datasetSchema, null, 2)}

Please suggest 5-10 derived metrics or features that might be useful for analysis. For each suggestion, provide:
1. A short name for the metric
2. A brief description of what it represents and why it's useful
3. The exact Python pandas code to calculate it

Format your response as a JSON array of objects with fields: name, description, and code.`;
        
        // Call the LLM API
        const response = await fetch("https://llmfoundry.straive.com/openai/v1/chat/completions", {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}:topcause-browser`
            },
            credentials: "include",
            body: JSON.stringify({
                model: "gpt-4o-mini",
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: userPrompt }
                ],
            }),
        });

        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }

        const data = await response.json();
        
        // Extract the content from the response
        const content = data.choices[0].message.content;
        
        // Try to parse the JSON from the content
        let suggestedMetrics;
        try {
            // Look for JSON array in the response
            const jsonMatch = content.match(/\[[\s\S]*\]/);
            if (jsonMatch) {
                suggestedMetrics = JSON.parse(jsonMatch[0]);
            } else {
                throw new Error("No JSON array found in response");
            }
        } catch (e) {
            console.error("Failed to parse JSON from LLM response:", e);
            console.log("Raw response:", content);
            
            // Fallback: Try to extract structured data from text response
            suggestedMetrics = extractMetricsFromText(content);
        }
        
        // Display the suggested metrics
        displayDerivedMetrics(suggestedMetrics, metricsContainer);
        
        // Store the derived metrics for later use
        derivedMetrics = suggestedMetrics;
        
    } catch (error) {
        metricsContainer.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        console.error(error);
    }
}

// Helper function to extract metrics from text if JSON parsing fails
function extractMetricsFromText(text) {
    const metrics = [];
    const sections = text.split(/\d+\.\s/).filter(Boolean);
    
    for (const section of sections) {
        try {
            const nameMatch = section.match(/(?:Name|Metric):\s*(.+?)(?:\n|$)/i);
            const descMatch = section.match(/Description:\s*(.+?)(?:\n\n|\n[A-Z]|$)/is);
            const codeMatch = section.match(/```python\s*([\s\S]*?)```/);
            
            if (nameMatch && descMatch && codeMatch) {
                metrics.push({
                    name: nameMatch[1].trim(),
                    description: descMatch[1].trim(),
                    code: codeMatch[1].trim()
                });
            }
        } catch (e) {
            console.error("Error parsing section:", e);
        }
    }
    
    return metrics;
}

// Function to display derived metrics and allow selection
function displayDerivedMetrics(metrics, container) {
    container.innerHTML = '';
    
    if (!metrics || metrics.length === 0) {
        container.innerHTML = '<div class="alert alert-warning">No derived metrics suggestions available.</div>';
        return;
    }
    
    // Create a form with checkboxes for each metric
    const form = document.createElement('form');
    form.id = 'derivedMetricsForm';
    
    metrics.forEach((metric, index) => {
        const card = document.createElement('div');
        card.className = 'card mb-3';
        
        const cardBody = document.createElement('div');
        cardBody.className = 'card-body';
        
        const header = document.createElement('div');
        header.className = 'd-flex justify-content-between align-items-center';
        
        const title = document.createElement('h5');
        title.className = 'card-title';
        title.textContent = metric.name;
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `metric-${index}`;
        checkbox.className = 'form-check-input';
        checkbox.checked = true; // Default to checked
        
        header.appendChild(title);
        header.appendChild(checkbox);
        
        const description = document.createElement('p');
        description.className = 'card-text';
        description.textContent = metric.description;
        
        const codeBlock = document.createElement('pre');
        codeBlock.className = 'p-2 mt-2';
        codeBlock.textContent = metric.code;
        
        cardBody.appendChild(header);
        cardBody.appendChild(description);
        cardBody.appendChild(codeBlock);
        card.appendChild(cardBody);
        form.appendChild(card);
    });
    
    container.appendChild(form);
    
    // Add derived metrics to the target column dropdown
    addDerivedMetricsToDropdown(metrics);
}

// Function to add derived metrics to the target column dropdown
function addDerivedMetricsToDropdown(metrics) {
    const targetDropdown = document.getElementById("targetColumn");
    
    // Create an optgroup for derived metrics
    let optgroup = targetDropdown.querySelector('optgroup[label="Derived Metrics"]');
    
    // If the optgroup doesn't exist, create it
    if (!optgroup) {
        optgroup = document.createElement('optgroup');
        optgroup.label = "Derived Metrics";
        targetDropdown.appendChild(optgroup);
    } else {
        // Clear existing derived metrics options
        optgroup.innerHTML = '';
    }
    
    // Add each metric to the optgroup
    metrics.forEach((metric, index) => {
        const option = document.createElement('option');
        option.value = `derived_${index}`;
        option.textContent = metric.name;
        option.dataset.metricIndex = index;
        optgroup.appendChild(option);
    });
}

// Function to run the TopCause analysis
async function runAnalysis() {
    if (!csvData) {
        alert("Please upload and preview a CSV file first.");
        return;
    }

    const targetDropdownEl = document.getElementById("targetColumn");
    const targetColumn = targetDropdownEl.value;
    if (!targetColumn) {
        alert("Please select a target column.");
        return;
    }

    // Show loading spinner in the output section
    const outputElement = document.getElementById("output");
    outputElement.innerHTML = '';
    outputElement.appendChild(createLoadingSpinner("Loading required packages..."));
    document.getElementById("resultSection").classList.remove("hidden");

    try {
        // Load required packages
        await pyodide.loadPackage("micropip");
        await pyodide.loadPackage("scikit-learn");
        outputElement.innerHTML = '';
        outputElement.appendChild(createLoadingSpinner("Running analysis..."));

        // Load the TopCause class directly from the Python code
        const topCauseCode = `
import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np
import warnings


class TopCause(BaseEstimator):
    '''TopCause finds the single largest action to improve a performance metric.

    Parameters
    ----------
    max_p : float
        maximum allowed probability of error (default: 0.05)
    percentile : float
        ignore high-performing outliers beyond this percentile (default: 0.95)
    min_weight : int
        minimum samples in a group. Drop groups with fewer (default: 3)

    Returns
    -------
    result_ : DataFrame
        rows = features evaluated
        columns:
            value: best value for this feature,
            gain: improvement in y if feature = value
            p: probability that this feature does not impact y
            type: how this feature's impact was calculated (e.g. \`num\` or \`cat\`)
    '''

    def __init__(
        self,
        max_p: float = 0.05,
        percentile: float = 0.95,
        min_weight: float = None,
    ):
        self.min_weight = min_weight
        self.max_p = max_p
        self.percentile = percentile

    def fit(self, X, y, sample_weight=None):  # noqa - capital X is a sklearn convention
        '''Returns the top causes of y from among X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            n_samples = rows = number of observations.
            n_features = columns = number of drivers/causes.
        y : array-line of shape (n_samples)

        Returns
        -------
        self : object
            Returns the instance itself.
        '''
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)  # noqa N806 X can be in uppercase
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'X has {X.shape[0]} rows, but y has {y.shape[0]} rows')

        # If values contain ±Inf treat it an NaN
        with pd.option_context('mode.use_inf_as_na', True):
            # If sample weights are not give, treat it as 1 for each row.
            # If sample weights are NaN, treat it as 0.
            if sample_weight is None:
                sample_weight = y.notnull().astype(int)
                # If no weights are specified, each category must have at least 3 rows
                min_weight = 3 if self.min_weight is None else self.min_weight
            elif not isinstance(sample_weight, pd.Series):
                sample_weight = pd.Series(sample_weight)
            sample_weight.fillna(0)

            # Calculate summary stats
            n = sample_weight.sum()
            weighted_y = y * sample_weight
            mean = weighted_y.sum() / n
            var = ((y - mean) ** 2 * sample_weight).sum() / n

            # Calculate impact for every column consistently
            results = {}
            for column, series in X.items():
                # Ignore columns identical to y
                if (series == y).all():
                    warnings.warn(f'column {column}: skipped. Identical to y')

                # Process column as NUMERIC, ORDERED CATEGORICAL or CATEGORICAL based on dtype
                # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
                kind = series.dtype.kind

                # By default, assume that this column can't impact y
                result = results[column] = {
                    'value': np.nan,
                    'gain': np.nan,
                    'p': 1.0,
                    'type': kind,
                }

                # ORDERED CATEGORICAL if kind is signed or unsigned int
                # TODO: Currently, it's treated as numeric. Fix this based # of distinct ints.
                if kind in 'iu':
                    series = series.astype(float)
                    kind = 'f'

                # NUMERIC if kind is float
                if kind in 'f':
                    # Drop missing values, pairwise
                    pair = pd.DataFrame({'values': series, 'weight': sample_weight, 'y': y})
                    pair.dropna(inplace=True)
                    
                    # Check if all values are identical
                    if pair['values'].nunique() <= 1:
                        warnings.warn(f'column {column}: skipped. All values are identical')
                        continue
                    
                    # Run linear regression to see if y increases/decreases with column
                    # TODO: use weighted regression
                    from scipy.stats import linregress

                    reg = linregress(pair['values'], pair['y'])

                    # If slope is +ve, pick value at the 95th percentile
                    # If slope is -ve, pick value at the 5th percentile
                    pair = pair.sort_values('values', ascending=True)
                    top = np.interp(
                        self.percentile if reg.slope >= 0 else 1 - self.percentile,
                        pair['weight'].cumsum() / pair['weight'].sum(),
                        pair['values'],
                    )

                    # Predict the gain based on linear regression
                    gain = reg.slope * top + reg.intercept - mean
                    if gain > 0:
                        result.update(value=top, gain=gain, p=reg.pvalue, type='num')

                # CATEGORICAL if kind is boolean, object, str or unicode
                elif kind in 'bOSU':
                    # Group into a DataFrame with 3 columns {value, weight, mean}
                    #   value: Each row has every unique value in the column
                    #   weight: Sum of sample_weights in each group
                    #   mean: mean(y) in each group, weighted by sample_weights
                    group = (
                        pd.DataFrame(
                            {'values': series, 'weight': sample_weight, 'weighted_y': weighted_y}
                        )
                        .dropna()
                        .groupby('values', sort=False)
                        .sum()
                    )
                    group['mean'] = group['weighted_y'] / group['weight']

                    # Pick the groups with highest mean(y), at >=95th percentile (or whatever).
                    # Ensure each group has at least min_weight samples.
                    group.sort_values('mean', inplace=True, ascending=True)
                    best_values = group.dropna(subset=['mean'])[
                        (group['weight'].cumsum() / group['weight'].sum() >= self.percentile)
                        & (group['weight'] >= min_weight)
                    ]

                    # If there's at least 1 group over 95th percentile with enough weights...
                    if len(best_values):
                        # X[series == top] is the largest group (by weights) above the 95th pc
                        top = best_values.sort_values('weight').iloc[-1]
                        gain = top['mean'] - mean
                        # Only consider positive gains
                        if gain > 0:
                            # Calculate p value using Welch test: scipy.stats.mstats.ttest_ind()
                            # https://en.wikipedia.org/wiki/Welch%27s_t-test
                            # github.com/scipy/scipy/blob/v1.5.4/scipy/stats/mstats_basic.py
                            subset = series == top.name
                            subseries = y[subset]
                            submean, subn = subseries.mean(), sample_weight[subset].sum()
                            with np.errstate(divide='ignore', invalid='ignore'):
                                diff = subseries - submean
                                vn1 = (diff**2 * sample_weight[subset]).sum() / subn
                                vn2 = var / n
                                df = (vn1 + vn2) ** 2 / (vn1**2 / (subn - 1) + vn2**2 / (n - 1))
                            df = 1 if np.isnan(df) else df
                            with np.errstate(divide='ignore', invalid='ignore'):
                                t = gain / (vn1 + vn2) ** 0.5
                            import scipy.special as special

                            p = special.betainc(0.5 * df, 0.5, df / (df + t * t))
                            # Update the result
                            result.update(value=top.name, gain=gain, p=p, type='cat')
                # WARN if kind is complex, timestamp, datetime, etc
                else:
                    warnings.warn(f'column {column}: unknown type {kind}')

            results = pd.DataFrame(results).T
            results.loc[results['p'] > self.max_p, ('value', 'gain')] = np.nan
            self.result_ = results.sort_values('gain', ascending=False)

        return self
`;

        // Load the TopCause code into Pyodide
        await pyodide.runPythonAsync(topCauseCode);

        // Get selected derived metrics
        const selectedMetrics = [];
        if (derivedMetrics && derivedMetrics.length > 0) {
            const form = document.getElementById('derivedMetricsForm');
            if (form) {
                derivedMetrics.forEach((metric, index) => {
                    const checkbox = document.getElementById(`metric-${index}`);
                    if (checkbox && checkbox.checked) {
                        selectedMetrics.push(metric);
                    }
                });
            }
        }

        // Check if the target is a derived metric
        let targetMetric = null;
        let actualTargetColumn = targetColumn;
        
        if (targetColumn.startsWith('derived_')) {
            const metricIndex = parseInt(targetColumn.split('_')[1]);
            if (metricIndex >= 0 && metricIndex < derivedMetrics.length) {
                targetMetric = derivedMetrics[metricIndex];
                // Add this metric to selected metrics if not already there
                if (!selectedMetrics.some(m => m.name === targetMetric.name)) {
                    selectedMetrics.push(targetMetric);
                }
            } else {
                throw new Error(`Invalid derived metric index: ${metricIndex}`);
            }
        }

        // Now run the analysis with the directly loaded TopCause class
        let script = `
import pandas as pd
from io import StringIO
import numpy as np
import json
import sys
import traceback

# Add debug print function that will show in the console
def debug_print(msg):
    print(f"DEBUG: {msg}", file=sys.stderr)

debug_print("Starting analysis")
csv_data = """${csvData}"""
debug_print("Loading CSV data")
df = pd.read_csv(StringIO(csv_data))
debug_print(f"DataFrame loaded with shape {df.shape}")

# Add derived metrics if any are selected
selected_metrics = ${JSON.stringify(selectedMetrics)}
debug_print(f"Selected metrics: {len(selected_metrics)}")

# Dictionary to map derived_X identifiers to actual column names
derived_column_map = {}

if selected_metrics:
    debug_print("Adding derived metrics")
    for i, metric in enumerate(selected_metrics):
        try:
            debug_print(f"Adding metric: {metric['name']}")
            
            # Get columns before executing the code
            before_cols = set(df.columns)
            
            # Execute the code to create the derived metric
            exec(metric['code'])
            
            # Get columns after executing the code
            after_cols = set(df.columns)
            
            # Find new columns created by the metric
            new_cols = after_cols - before_cols
            
            if new_cols:
                # Use the first new column as the derived column
                derived_col = list(new_cols)[0]
                derived_column_map[f'derived_{i}'] = derived_col
                debug_print(f"Mapped derived_{i} to {derived_col}")
            else:
                # Try to use the metric name as a fallback
                fallback_name = metric['name'].lower().replace(' ', '_')
                if fallback_name in df.columns:
                    derived_column_map[f'derived_{i}'] = fallback_name
                    debug_print(f"Mapped derived_{i} to {fallback_name} (fallback)")
                else:
                    debug_print(f"Warning: Could not determine column for metric {metric['name']}")
            
            debug_print(f"Successfully added metric: {metric['name']}")
        except Exception as e:
            debug_print(f"Error adding metric {metric['name']}: {str(e)}")
            debug_print(traceback.format_exc())

# Check if target is a derived metric
target_column = "${actualTargetColumn}"
debug_print(f"Original target column: {target_column}")

# If the target is a derived metric, get the actual column name
if target_column.startswith('derived_'):
    if target_column in derived_column_map:
        target_column = derived_column_map[target_column]
        debug_print(f"Resolved derived target to: {target_column}")
    else:
        # If we couldn't map it, try to extract from the metric directly
        target_metric = ${targetMetric ? JSON.stringify(targetMetric) : 'None'}
        if target_metric is not None:
            debug_print(f"Trying to resolve target from metric: {target_metric['name']}")
            
            # Get columns before executing the code
            before_cols = set(df.columns)
            
            # Execute the metric code again to be sure
            exec(target_metric['code'])
            
            # Get columns after executing the code
            after_cols = set(df.columns)
            
            # Find new columns created by the metric
            new_cols = after_cols - before_cols
            
            if new_cols:
                # Use the first new column as the target
                target_column = list(new_cols)[0]
                debug_print(f"Resolved target to newly created column: {target_column}")
            else:
                # Try to use the metric name as a fallback
                fallback_name = target_metric['name'].lower().replace(' ', '_')
                if fallback_name in df.columns:
                    target_column = fallback_name
                    debug_print(f"Resolved target to metric name: {target_column}")
                else:
                    # Last resort: look for columns that might match the metric name
                    possible_cols = [col for col in df.columns if fallback_name in col.lower()]
                    if possible_cols:
                        target_column = possible_cols[0]
                        debug_print(f"Resolved target to similar column: {target_column}")

# Verify the target column exists in the dataframe
if target_column not in df.columns:
    available_cols = ", ".join(df.columns)
    raise ValueError(f"Target column '{target_column}' not found in dataframe. Available columns: {available_cols}")

debug_print(f"Final target column: {target_column}")

class TopCauseModel:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.model = TopCause()

    def analyze(self):
        # Make a copy of the dataframe to avoid modifying the original
        X = self.df.copy()
        
        # Ensure the target column is numeric
        try:
            # Try to convert the target column to numeric if it's not already
            if not pd.api.types.is_numeric_dtype(X[self.target_column]):
                debug_print(f"Converting target column {self.target_column} to numeric")
                X[self.target_column] = pd.to_numeric(X[self.target_column], errors='coerce')
                # Fill NaN values with 0 to avoid issues
                X[self.target_column] = X[self.target_column].fillna(0)
        except Exception as e:
            debug_print(f"Error converting target column to numeric: {str(e)}")
            debug_print(traceback.format_exc())
        
        # Check if target column has variation
        if X[self.target_column].nunique() <= 1:
            raise ValueError(f"Target column '{self.target_column}' has no variation (all values are identical). Please select a different target column.")
        
        # Remove the target column from the features
        y = X[self.target_column]
        X = X.drop(columns=[self.target_column])
        
        # Convert all feature columns to appropriate types
        for col in X.columns:
            try:
                # Try to convert object columns to numeric if possible
                if X[col].dtype == 'object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='ignore')
                    except:
                        pass
                
                # Handle any remaining non-numeric columns
                if not pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_categorical_dtype(X[col]):
                    debug_print(f"Converting column {col} to categorical")
                    X[col] = X[col].astype('category')
                
                # Check for columns with no variation
                if X[col].nunique() <= 1:
                    debug_print(f"Column {col} has no variation (all values are identical)")
            except Exception as e:
                debug_print(f"Error processing column {col}: {str(e)}")
        
        debug_print(f"X shape: {X.shape}, y shape: {y.shape}")
        debug_print(f"Target column dtype: {y.dtype}")
        debug_print(f"Target column unique values: {y.nunique()}")
        
        # Fit the model
        self.model.fit(X, y)
        debug_print("Model fitted")
        return self.model.result_

try:
    # Load required packages for TopCause
    import scipy.stats
    import scipy.special
    debug_print("Packages loaded")

    # Create the model and run analysis
    debug_print(f"Target column: {target_column}")
    
    # Check if the target column is valid for analysis
    if target_column in df.columns:
        # Check if the column has valid numeric data
        try:
            # Try to convert to numeric to see if it's possible
            test_numeric = pd.to_numeric(df[target_column], errors='coerce')
            if test_numeric.isna().all():
                raise ValueError(f"Target column '{target_column}' cannot be converted to numeric values. Please select a different target.")
            
            # Check if there's variation in the target column
            if df[target_column].nunique() <= 1:
                raise ValueError(f"Target column '{target_column}' has no variation (all values are identical). Please select a different target column.")
            
            # If we have some numeric values, proceed with analysis
            top_cause = TopCauseModel(df, target_column)
            debug_print("Model created")
            result = top_cause.analyze()
            debug_print(f"Analysis complete, result type: {type(result)}")
            
            # Check if result is empty or None
            if result is None:
                debug_print("Result is None")
                print(json.dumps({"error": "Analysis returned None. There might be an issue with the data."}))
            elif hasattr(result, 'empty') and result.empty:
                debug_print("Result is empty DataFrame")
                print(json.dumps({"error": "Analysis produced no results. Try a different target column."}))
            else:
                debug_print(f"Result shape: {getattr(result, 'shape', 'No shape attribute')}")
                # Convert to JSON string for reliable return
                if isinstance(result, pd.DataFrame):
                    # Convert DataFrame to JSON
                    json_result = result.reset_index().to_json(orient='records')
                    print(json_result)
                else:
                    # For any other type, convert to string
                    print(json.dumps({"result": str(result)}))
        except Exception as e:
            debug_print(f"Error during analysis: {str(e)}")
            debug_print(traceback.format_exc())
            print(json.dumps({"error": str(e)}))
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
except Exception as e:
    import traceback
    debug_print(f"Exception: {str(e)}")
    debug_print(traceback.format_exc())
    print(json.dumps({"error": str(e)}))

# Always return something to ensure we don't get undefined
print(json.dumps({"status": "completed"}))
`;

        try {
            // Run the Python code and get the result
            // We need to capture the stdout from Python
            const stdout = [];
            pyodide.setStdout({
                write: (text) => {
                    stdout.push(text);
                }
            });
            
            await pyodide.runPythonAsync(script);
            
            // Join all stdout output
            const resultStr = stdout.join('');
            console.log("Raw Python output:", resultStr);

            // Try to find and parse the JSON array in the output
            try {
                // Look for JSON array in the output
                const startBracket = resultStr.indexOf('[');
                const endBracket = resultStr.lastIndexOf(']') + 1;
                
                if (startBracket >= 0 && endBracket > startBracket) {
                    // Extract just the JSON array part
                    const jsonArrayStr = resultStr.substring(startBracket, endBracket);
                    try {
                        const jsonArray = JSON.parse(jsonArrayStr);
                        displayResultsTable(jsonArray, outputElement);
                    } catch (parseError) {
                        outputElement.innerText = "Error parsing results: " + parseError.message;
                        console.error(parseError);
                    }
                } else {
                    // If no JSON array found, check if it's a JSON object
                    try {
                        const result = JSON.parse(resultStr);
                        if (result.error) {
                            outputElement.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
                        } else {
                            outputElement.innerText = JSON.stringify(result, null, 2);
                        }
                    } catch (e) {
                        // If all parsing fails, show the raw output
                        outputElement.innerText = resultStr;
                    }
                }
            } catch (error) {
                outputElement.innerText = "Error processing results: " + error.message;
                console.error(error);
            }
        } catch (error) {
            outputElement.innerText = "JavaScript Error: " + error.message;
            console.error(error);
        }
    } catch (error) {
        outputElement.innerText = "Error: " + error.message;
        console.error(error);
    }
}

// Add event listeners
document.getElementById("previewButton").addEventListener("click", previewCSV);
document.getElementById("analysisButton").addEventListener("click", runAnalysis);
document.getElementById("getDerivedMetricsButton").addEventListener("click", getDerivedMetricsSuggestions);

// Add this helper function at the end of the file
function displayResultsTable(data, container) {
    // Create a table to display the results
    const table = document.createElement('table');
    table.className = 'table table-striped table-hover';
    
    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    // Get all possible keys from the data
    const allKeys = new Set();
    data.forEach(item => {
        Object.keys(item).forEach(key => allKeys.add(key));
    });
    
    // Create header cells
    allKeys.forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    
    // Add data rows
    data.forEach(item => {
        const row = document.createElement('tr');
        
        allKeys.forEach(key => {
            const cell = document.createElement('td');
            // Format the value or show null/undefined as empty
            const value = item[key];
            if (value === null || value === undefined) {
                cell.textContent = '-';
                cell.className = 'text-muted';
            } else if (typeof value === 'number') {
                // Format numbers to 2 decimal places
                cell.textContent = Number.isInteger(value) ? value : value.toFixed(2);
            } else {
                cell.textContent = value;
            }
            row.appendChild(cell);
        });
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    
    // Clear the container and add the table
    container.innerHTML = '';
    container.appendChild(table);
}