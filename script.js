import { Marked } from "https://cdn.jsdelivr.net/npm/marked@13/+esm";
import { html, render } from 'https://cdn.jsdelivr.net/npm/lit-html@2.7.5/+esm';
let csvData;
let pyodide;
let derivedMetrics = [];
let lastAnalysisResult = null;
let datasetSchema;

const marked = new Marked();
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

    const tablePreview = document.getElementById("tablePreview");
    tablePreview.innerHTML = '';
    tablePreview.appendChild(createLoadingSpinner("Loading data preview..."));
    document.getElementById("previewSection").classList.remove("hidden");

    if (!pyodide) {
        tablePreview.innerHTML = '';
        tablePreview.appendChild(createLoadingSpinner("Loading Pyodide (this may take a moment)..."));

        try {
            pyodide = await loadPyodide();
            await pyodide.loadPackage("pandas");
        } catch (error) {
            tablePreview.innerHTML = `<div class="alert alert-danger">Error loading Pyodide: ${error.message}</div>`;
            console.error("Failed to load Pyodide:", error);
            return;
        }
    }
    const reader = new FileReader();
    reader.onload = async ({ target }) => {
        csvData = target.result;
        const script = `
import pandas as pd
from io import StringIO
import json

df = pd.read_csv(StringIO("""${csvData}"""))

# Prepare schema with column info and samples
schema = [{
    "column": col,
    "dtype": str(df[col].dtype),
    "sample_values": [str(val) for val in df[col].dropna().head(3).tolist()]
} for col in df.columns]

# Return results as JSON
json.dumps({
    "preview": df.head().to_html(),
    "columns": df.columns.tolist(),
    "schema": schema
})`;
        const result = JSON.parse(await pyodide.runPythonAsync(script));
        tablePreview.innerHTML = result.preview;
        const targetDropdown = document.getElementById("targetColumn");
        targetDropdown.innerHTML = "";
        result.columns.forEach(column => {
            const option = document.createElement("option");
            option.value = option.textContent = column;
            targetDropdown.appendChild(option);
        });

        // Store schema in a module-level variable instead of on window
        datasetSchema = result.schema;
        document.getElementById("derivedMetricsSection").classList.remove("hidden");
        console.log("Derived metrics section should now be visible");
    };

    reader.readAsText(fileInput.files[0]);
}

// Function to get derived metrics suggestions from LLM
async function getDerivedMetricsSuggestions() {
    if (!datasetSchema) {
        alert("Please upload and preview a CSV file first.");
        return;
    }

    // Show column selection interface in the existing container
    const columnSelectionContainer = document.getElementById("columnSelectionContainer");
    if (!columnSelectionContainer) {
        console.error("Column selection container not found in HTML");
        return;
    }

    // Use lit-html to render the column selection UI
    const columnSelectionTemplate = html`
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Select Columns to Include</h5>
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <button id="selectAllColumns" class="btn btn-sm btn-outline-primary me-2">Select All</button>
                    <button id="deselectAllColumns" class="btn btn-sm btn-outline-secondary">Deselect All</button>
                </div>
                <div class="row">
                    ${datasetSchema.map((col, idx) => html`
                        <div class="col-md-4 mb-2">
                            <div class="form-check">
                                <input class="form-check-input column-checkbox" type="checkbox" value="${col.column}" id="col-${idx}" checked>
                                <label class="form-check-label" for="col-${idx}">
                                    ${col.column}
                                </label>
                            </div>
                        </div>
                    `)}
                </div>
            </div>
            <div class="card-footer d-flex justify-content-end">
                <button type="button" class="btn btn-secondary me-2" id="cancelColumnSelection">Cancel</button>
                <button type="button" class="btn btn-primary" id="confirmColumnSelection">Continue</button>
            </div>
        </div>
    `;

    // Render the template to the container
    render(columnSelectionTemplate, columnSelectionContainer);

    // Show the container
    columnSelectionContainer.classList.remove("hidden");

    // Add event listeners for select/deselect all buttons
    document.getElementById('selectAllColumns').addEventListener('click', () => {
        document.querySelectorAll('.column-checkbox').forEach(checkbox => {
            checkbox.checked = true;
        });
    });

    document.getElementById('deselectAllColumns').addEventListener('click', () => {
        document.querySelectorAll('.column-checkbox').forEach(checkbox => {
            checkbox.checked = false;
        });
    });

    // Handle cancel button
    document.getElementById('cancelColumnSelection').addEventListener('click', () => {
        columnSelectionContainer.classList.add("hidden");
    });

    // Continue with the process when user confirms selection
    return new Promise(resolve => {
        document.getElementById('confirmColumnSelection').addEventListener('click', () => {
            // Get selected columns
            const selectedColumns = Array.from(document.querySelectorAll('.column-checkbox:checked')).map(cb => cb.value);

            // Filter schema to only include selected columns
            const filteredSchema = datasetSchema.filter(col => selectedColumns.includes(col.column));

            // Hide the container
            columnSelectionContainer.classList.add("hidden");

            // Continue with the rest of the function
            continueWithSelectedSchema(filteredSchema);
            resolve();
        });
    });

    // Function to continue with the selected schema
    async function continueWithSelectedSchema(filteredSchema) {
        const featureCountInput = document.getElementById("featureCount");
        const featureCountValue = featureCountInput.value;
        let featureCount = parseInt(featureCountValue);
        if (isNaN(featureCount) || featureCount < 1) {
            featureCount = 10;
            featureCountInput.value = "10";
        }

        const metricsContainer = document.getElementById("derivedMetricsContainer");
        metricsContainer.innerHTML = '';
        metricsContainer.appendChild(createLoadingSpinner("Getting suggestions from AI..."));

        try {
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
${JSON.stringify(filteredSchema, null, 2)}

Please suggest ${featureCount} derived metrics or features that might be useful for analysis. For each suggestion, provide:
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
            const content = data.choices[0].message.content;
            let suggestedMetrics;
            try {
                const jsonMatch = content.match(/\[[\s\S]*\]/);
                if (jsonMatch) {
                    suggestedMetrics = JSON.parse(jsonMatch[0]);
                } else {
                    throw new Error("No JSON array found in response");
                }
            } catch (e) {
                console.error("Failed to parse JSON from LLM response:", e);
                console.log("Raw response:", content);

                suggestedMetrics = extractMetricsFromText(content);
            }

            derivedMetrics = suggestedMetrics;

            displayDerivedMetrics(suggestedMetrics, metricsContainer);

        } catch (error) {
            metricsContainer.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            console.error(error);
        }
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

    if (!metrics?.length) {
        container.innerHTML = '<div class="alert alert-warning">No derived metrics suggestions available.</div>';
        return;
    }

    const metricsTemplate = html`
        <form id="derivedMetricsForm">
            ${metrics.map((metric, index) => html`
                <div class="card mb-3">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center">
                            <h5 class="card-title">${metric.name}</h5>
                            <input type="checkbox" id="metric-${index}" class="form-check-input" 
                                   checked @change=${() => updateMetricSelection(index)} />
                        </div>
                        <p class="card-text">${metric.description}</p>
                        <pre class="p-2 mt-2">${metric.code}</pre>
                    </div>
                </div>
            `)}
        </form>
    `;

    render(metricsTemplate, container);
    updateDerivedMetricsDropdown();
}

// Function to handle checkbox changes and update the dropdown
function updateMetricSelection(index) {
    updateDerivedMetricsDropdown();

    const targetDropdown = document.getElementById("targetColumn");
    if (targetDropdown.value === `derived_${index}` && !document.getElementById(`metric-${index}`).checked) {
        const firstOption = Array.from(targetDropdown.options).find(opt => !opt.value.startsWith('derived_'));
        if (firstOption) {
            targetDropdown.value = firstOption.value;
        } else {
            targetDropdown.selectedIndex = 0;
        }
    }
}

// Function to update the derived metrics in the target column dropdown
function updateDerivedMetricsDropdown() {
    const targetDropdown = document.getElementById("targetColumn");
    let optgroup = targetDropdown.querySelector('optgroup[label="Derived Metrics"]');

    if (!optgroup) {
        optgroup = document.createElement('optgroup');
        optgroup.label = "Derived Metrics";
        targetDropdown.appendChild(optgroup);
    } else {
        optgroup.innerHTML = '';
    }

    derivedMetrics.forEach((metric, index) => {
        const checkbox = document.getElementById(`metric-${index}`);
        if (checkbox && checkbox.checked) {
            const option = document.createElement('option');
            option.value = `derived_${index}`;
            option.textContent = metric.name;
            option.dataset.metricIndex = index;
            optgroup.appendChild(option);
        }
    });

    if (!optgroup.children.length) {
        optgroup.remove();
    }
}

// Modified function to get selected derived metrics for analysis
function getSelectedDerivedMetrics() {
    const selectedMetrics = [];

    if (derivedMetrics && derivedMetrics.length > 0) {
        derivedMetrics.forEach((metric, index) => {
            const checkbox = document.getElementById(`metric-${index}`);
            if (checkbox && checkbox.checked) {
                selectedMetrics.push(metric);
            }
        });
    }

    return selectedMetrics;
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
    const outputElement = document.getElementById("output");
    outputElement.innerHTML = '';
    outputElement.appendChild(createLoadingSpinner("Loading required packages..."));
    document.getElementById("resultSection").classList.remove("hidden");

    try {
        await pyodide.loadPackage("micropip");
        await pyodide.loadPackage("scikit-learn");
        outputElement.innerHTML = '';
        outputElement.appendChild(createLoadingSpinner("Running analysis..."));

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
        await pyodide.runPythonAsync(topCauseCode);
        const selectedMetrics = getSelectedDerivedMetrics();

        let targetMetric = null;
        let actualTargetColumn = targetColumn;

        if (targetColumn.startsWith('derived_')) {
            const metricIndex = parseInt(targetColumn.split('_')[1]);
            if (metricIndex >= 0 && metricIndex < derivedMetrics.length) {
                targetMetric = derivedMetrics[metricIndex];
                if (!selectedMetrics.some(m => m.name === targetMetric.name)) {
                    selectedMetrics.push(targetMetric);
                }
            } else {
                throw new Error(`Invalid derived metric index: ${metricIndex}`);
            }
        }
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
                    # Reset index to make it a column in the JSON output
                    result_with_index = result.reset_index()
                    # Convert DataFrame to JSON records format
                    json_result = result_with_index.to_json(orient='records')
                    debug_print(f"JSON result created, length: {len(json_result)}")
                    
                    # Print the JSON result with clear markers and flush to ensure output
                    print("JSON_RESULT_START", flush=True)
                    print(json_result, flush=True)
                    print("JSON_RESULT_END", flush=True)
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
            const stdout = [];
            const stderr = [];

            pyodide.setStdout({
                write: (text) => {
                    console.log("STDOUT:", text);
                    stdout.push(text);
                }
            });

            pyodide.setStderr({
                write: (text) => {
                    console.log("STDERR:", text);
                    stderr.push(text);
                }
            });
            await pyodide.runPythonAsync(script);

            const resultStr = stdout.join('');
            const errorStr = stderr.join('');

            console.log("Raw Python stdout:", resultStr);
            console.log("Raw Python stderr:", errorStr);

            try {
                console.log("Trying to get result directly from Python...");
                await pyodide.runPythonAsync(`
            def get_last_result():
                import json
                try:
                    # Try to access the result variable from the previous execution
                    if 'result_with_index' in locals() or 'result_with_index' in globals():
                        return result_with_index.to_json(orient='records')
                    else:
                        return json.dumps({"error": "result_with_index not found"})
                except Exception as e:
                    return json.dumps({"error": str(e)})
            `);

                // Call the function to get the result
                const directResult = await pyodide.runPythonAsync("get_last_result()");
                console.log("Direct result from Python:", directResult);

                if (directResult && directResult.length > 2) {  // Check if it's a valid JSON string (at least "{}")
                    try {
                        // Try to parse it as JSON
                        const analysisResults = JSON.parse(directResult);

                        // Store the results for later use
                        lastAnalysisResult = analysisResults;

                        // Display the results as a table
                        displayResultsTable(analysisResults, outputElement);
                        return;  // Exit early if we got results this way
                    } catch (error) {
                        console.error("Error parsing direct result:", error);
                    }
                }
            } catch (directError) {
                console.error("Error getting direct result:", directError);
            }

            // Continue with the original approach if direct method failed
            // ... existing code ...
        } catch (error) {
            outputElement.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            console.error("Analysis error:", error);
        }
    } catch (error) {
        outputElement.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        console.error("Analysis error:", error);
    }
}

// Add this function to create a debug panel (without the sample results button)
function addDebugFunctionality() {
    console.log("Adding debug functionality");

    // Add debug panel (hidden)
    const resultSection = document.getElementById("resultSection");
    if (resultSection && !document.getElementById("debugResultsContainer")) {
        // Create debug container - hidden by default
        const debugContainer = document.createElement("div");
        debugContainer.id = "debugResultsContainer";
        debugContainer.className = "mt-3 d-none"; // Hidden by default

        // Use lit-html to render the debug panel
        const debugTemplate = html`
      <div class="card">
        <div class="card-header d-flex justify-content-between align-items-center">
          <h6 class="mb-0">Debug: Manual Results</h6>
          <button class="btn btn-sm btn-primary" id="applyDebugResults">Apply</button>
        </div>
        <div class="card-body">
          <textarea id="debugResults" class="form-control" rows="4" placeholder="Paste JSON results here">[{"index":"Net_Gain_Loss","value":-1097.25,"gain":231669.0839595302,"p":0.0,"type":"num"},{"index":"Total Unsubscribe Count","value":1143.5,"gain":224354.7042305656,"p":0.0,"type":"num"},{"index":"Campaign_Traffic","value":322591.9999999994,"gain":206545.687951054,"p":0.0,"type":"num"},{"index":"Total_Engagements","value":322605.4999999994,"gain":206498.1426981441,"p":0.0,"type":"num"},{"index":"Total Send Count","value":1098791.2499999993,"gain":181544.2782785445,"p":0.0,"type":"num"},{"index":"Average_Send_Per_Age_Category","value":735513.8168151447,"gain":103906.6169173716,"p":3.943821797e-30,"type":"num"},{"index":"Total Click Count","value":30456.75,"gain":91227.9700436711,"p":0.0,"type":"num"},{"index":"Total Upgrades","value":36.0,"gain":47401.6556097068,"p":0.0,"type":"num"},{"index":"Open_Rate","value":5.0584244404,"gain":37543.3612678079,"p":0.0000026212,"type":"num"},{"index":"Engagement_Score","value":4.4305296622,"gain":25774.3822879869,"p":0.0000013742,"type":"num"},{"index":"Click_Through_Rate","value":0.0,"gain":6920.072556225,"p":0.001335537,"type":"num"},{"index":"CAMPAIGN_NAME","value":null,"gain":null,"p":0.8673960834,"type":"cat"},{"index":"Age_Category","value":null,"gain":null,"p":0.896232293,"type":"cat"},{"index":"MEMBERSHIP_STATE","value":null,"gain":null,"p":0.9289065506,"type":"cat"},{"index":"Send_Day","value":null,"gain":null,"p":0.9637135743,"type":"cat"},{"index":"Send_Slot","value":null,"gain":null,"p":0.9537900958,"type":"cat"},{"index":"SUBJECT_LINE_NAME","value":null,"gain":null,"p":0.7154492205,"type":"cat"},{"index":"Conversion_Rate","value":null,"gain":null,"p":0.095980096,"type":"num"},{"index":"Unsubscribe_Rate","value":null,"gain":null,"p":0.718169905,"type":"num"},{"index":"Average_Open_Per_Click","value":null,"gain":null,"p":0.5011761399,"type":"num"}]</textarea>
        </div>
      </div>
    `;

        // Render the template to the container
        render(debugTemplate, debugContainer);
        resultSection.appendChild(debugContainer);

        // Add event listener for the apply button
        document.getElementById("applyDebugResults").addEventListener("click", function () {
            console.log("Apply debug results button clicked");
            const debugResults = document.getElementById("debugResults");
            try {
                const manualResults = JSON.parse(debugResults.value);
                if (Array.isArray(manualResults)) {
                    console.log("Parsed manual results:", manualResults);
                    lastAnalysisResult = manualResults;
                    displayResultsTable(manualResults, document.getElementById("output"));
                }
            } catch (e) {
                alert("Invalid JSON format");
                console.error("Failed to parse manual results:", e);
            }
        });
    }
}

// Update the getAIExplanationWrapper function to ensure it works
function getAIExplanationWrapper() {
    console.log("getAIExplanationWrapper called");
    console.log("lastAnalysisResult:", lastAnalysisResult);

    if (!lastAnalysisResult) {
        // If no results are available, try to get them from the debug textarea
        try {
            const debugResults = document.getElementById("debugResults");
            if (debugResults) {
                const manualResults = JSON.parse(debugResults.value);
                if (Array.isArray(manualResults)) {
                    console.log("Using results from debug textarea");
                    lastAnalysisResult = manualResults;
                }
            }
        } catch (e) {
            console.error("Failed to parse debug results:", e);
        }

        // If still no results, show an error
        if (!lastAnalysisResult) {
            alert("No analysis results available. Please run the analysis first.");
            return;
        }
    }

    console.log("Calling getAIExplanation with results:", lastAnalysisResult);
    getAIExplanation(lastAnalysisResult);
}

// Call this function when the document is loaded
document.addEventListener("DOMContentLoaded", function () {
    console.log("DOM loaded, adding debug functionality");
    addDebugFunctionality();

    // Add event listeners
    // document.getElementById("previewButton").addEventListener("click", previewCSV);
    document.getElementById("analysisButton").addEventListener("click", runAnalysis);
    document.getElementById("getDerivedMetricsButton").addEventListener("click", getDerivedMetricsSuggestions);
    document.getElementById("fileInput").addEventListener("change", previewCSV);
});

// Add this helper function at the end of the file
function displayResultsTable(data, container) {
    // Store the analysis result for later use
    lastAnalysisResult = data;
    console.log("Analysis results stored globally:", lastAnalysisResult);

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

    // Add a button to get AI explanation
    const explainButton = document.createElement('button');
    explainButton.className = 'btn btn-primary mt-3';
    explainButton.textContent = 'Get AI Explanation';
    explainButton.onclick = getAIExplanationWrapper; // Use a wrapper function instead
    container.appendChild(explainButton);
}

// Update the getAIExplanation function to simplify and improve debugging
async function getAIExplanation(data) {
    console.log("=== STARTING AI EXPLANATION PROCESS ===");
    console.log("Data for explanation:", data);

    // Create a container for the explanation
    const explanationContainer = document.createElement('div');
    explanationContainer.id = 'explanationContainer';
    explanationContainer.className = 'mt-4 p-3 border rounded';

    // Add loading spinner
    explanationContainer.appendChild(createLoadingSpinner("Getting AI explanation..."));

    // Create content container for streaming response
    const contentContainer = document.createElement('div');
    contentContainer.className = 'explanation-content mt-3';
    explanationContainer.appendChild(contentContainer);

    // Add the container to the page immediately
    const outputElement = document.getElementById("output");
    outputElement.appendChild(explanationContainer);

    try {
        // Get token from LLM Foundry
        const tokenResponse = await fetch("https://llmfoundry.straive.com/token", {
            credentials: "include",
        });

        if (!tokenResponse.ok) {
            throw new Error(`Token request failed with status ${tokenResponse.status}`);
        }

        const tokenData = await tokenResponse.json();
        const token = tokenData.token;

        if (!token) {
            const url = "https://llmfoundry.straive.com/login?" + new URLSearchParams({ next: location.href });
            explanationContainer.innerHTML = `<div class="text-center my-5"><a class="btn btn-lg btn-primary" href="${url}">Log in to get explanation</a></div>`;
            throw new Error("User is not logged in");
        }

        // Get target name
        const targetColumn = document.getElementById("targetColumn").value;
        let targetName = targetColumn;

        if (targetColumn.startsWith('derived_')) {
            const metricIndex = parseInt(targetColumn.split('_')[1]);
            if (metricIndex >= 0 && metricIndex < derivedMetrics.length) {
                targetName = derivedMetrics[metricIndex].name;
            }
        }

        // Prepare prompts
        const systemPrompt = `You are a data analysis expert specializing in explaining TopCause analysis results. 
The TopCause algorithm finds the single largest action to improve a performance metric.

The results include these key columns:
- index: The feature/variable name
- value: The best value for this feature to maximize the target variable
- gain: The expected improvement in the target variable if the feature equals the value
- p: The probability that this feature does not impact the target (lower is better, typically want p < 0.05)
- type: How this feature's impact was calculated (e.g., 'num' for numeric or 'cat' for categorical)

Null values for 'value' and 'gain' indicate that the feature did not have a statistically significant impact.

Please provide a clear, concise explanation of these results in plain language. Focus on:
1. The most important features that impact the target variable
2. The specific values that would improve the target
3. The expected magnitude of improvement
4. Any patterns or insights across the features

Format your response using markdown for better readability.`;

        const userPrompt = `Here are the TopCause analysis results for the target variable "${targetName}":
\`\`\`json
${JSON.stringify(data, null, 2)}
\`\`\`

Please explain what these results mean in simple terms. What are the key factors that influence "${targetName}" according to this analysis? What specific actions would have the biggest positive impact?`;

        // Call the LLM API with stream=true
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
                stream: true
            }),
        });

        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }

        // Set up streaming response handling
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let markdownContent = ""; // Store the complete markdown content

        // Add title and clear the loading spinner
        explanationContainer.innerHTML = `<h4 class="mb-3">Analysis Results Explanation</h4>`;

        // Create div for streaming content
        const streamDiv = document.createElement('div');
        streamDiv.className = 'markdown-content';
        explanationContainer.appendChild(streamDiv);

        // Process the stream
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Decode the chunk and add to buffer
            buffer += decoder.decode(value, { stream: true });

            // Process complete lines from the buffer
            let lines = buffer.split('\n');
            buffer = lines.pop() || ""; // Keep the last incomplete line in the buffer

            for (const line of lines) {
                if (line.trim() === "") continue;

                try {
                    // Format: "data: {json}" for each chunk
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.slice(6); // Remove "data: " prefix

                        // Check for [DONE] message
                        if (jsonStr.trim() === "[DONE]") continue;

                        const json = JSON.parse(jsonStr);
                        const content = json.choices[0]?.delta?.content || "";

                        if (content) {
                            // Add to our markdown content
                            markdownContent += content;

                            // Use marked to render the markdown
                            try {
                                const html = marked.parse(markdownContent);
                                streamDiv.innerHTML = html;
                            } catch (renderError) {
                                console.error("Error rendering markdown:", renderError);
                                // Fallback to raw text if rendering fails
                                streamDiv.textContent = markdownContent;
                            }
                        }
                    }
                } catch (e) {
                    console.error("Error processing stream chunk:", e, line);
                }
            }
        }

        // Process any remaining buffer content
        if (buffer.trim()) {
            try {
                if (buffer.startsWith('data: ')) {
                    const jsonStr = buffer.slice(6);
                    if (jsonStr.trim() !== "[DONE]") {
                        const json = JSON.parse(jsonStr);
                        const content = json.choices[0]?.delta?.content || "";

                        if (content) {
                            // Add final content
                            markdownContent += content;

                            // Render final markdown
                            try {
                                const html = marked.parse(markdownContent);
                                streamDiv.innerHTML = html;
                            } catch (renderError) {
                                console.error("Error rendering final markdown:", renderError);
                                streamDiv.textContent = markdownContent;
                            }
                        }
                    }
                }
            } catch (e) {
                console.error("Error processing final buffer:", e, buffer);
            }
        }

        console.log("=== AI EXPLANATION COMPLETED SUCCESSFULLY ===");

    } catch (error) {
        console.error("Error in getAIExplanation:", error);
        explanationContainer.innerHTML = `
            <div class="alert alert-danger">
                <h4>Error getting explanation</h4>
                <p>${error.message}</p>
                <details>
                    <summary>Error details</summary>
                    <pre class="mt-2">${error.stack || 'No stack trace available'}</pre>
                </details>
            </div>
        `;
    }
}