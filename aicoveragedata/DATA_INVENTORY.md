# AI Coverage Data Inventory

This is the detailed project map. The root `README.md` is the quick start.

## Folder Map

| Folder | Kind | Purpose |
| --- | --- | --- |
| `agent/` | Source | DSPy chatbot package. |
| `agent/core/` | Source | Main agent flow and environment config. |
| `agent/context/` | Source | Dataset, regression, code, and HTML context readers. |
| `agent/llm/` | Source | DSPy module and fallback system prompt. |
| `agent/memory/` | Source | Chat history storage. |
| `agent/interface/` | Source | CLI entrypoint. |
| `agent/state/` | Runtime | Ignored chat history and DSPy cache. |
| `app/` | Source | Site generator, server, chatbot widget. |
| `data/` | Source | Dataset loader and coverage summary. |
| `charts/` | Source | Standalone chart scripts and generated chart-page builders. |
| `charts/exploratory/` | Source | Older matplotlib chart experiments. |
| `charts/pages/` | Source | Standalone generated HTML chart page builders. |
| `regression/` | Source | Regression diagnostics, report models, report page. |
| `regression/report/` | Source | Active regression report implementation. |
| `site/` | Generated | HTML pages, CSV downloads, reports. |

## Source Files

### `app/`

| File | Purpose |
| --- | --- |
| `app/dashboard.py` | Generates the dashboard, profile pages, and regression report. |
| `app/dashboard_server.py` | Serves `site/` locally and handles `/api/agent`. |
| `app/agent_widget.py` | Floating AI chat tab, controls, messages, clear/history behavior. |

### `agent/`

| File | Purpose |
| --- | --- |
| `agent/core/agent.py` | Main answer path, DSPy/fallback routing, memory write. |
| `agent/core/config.py` | Loads `.env`, model settings, memory limits, DSPy flags. |
| `agent/context/data_catalog.py` | Central file paths for dashboard and regression outputs. |
| `agent/context/tools.py` | Builds grounded context from CSVs, code, HTML, and chat history. |
| `agent/llm/dspy_agent.py` | DSPy signature/module using history, question, and local context. |
| `agent/llm/prompts.py` | Fallback prompt for direct Responses API mode. |
| `agent/memory/history.py` | Reads, writes, formats, and clears chat sessions. |
| `agent/interface/cli.py` | Command-line chatbot. |
| `agent/requirements.txt` | Python dependencies. |
| `agent/README.md` | Agent-specific setup notes. |

### `regression/`

| File | Purpose |
| --- | --- |
| `regression/baseline.py` | Shared split, scaler pipeline, metrics, and coefficient helpers. |
| `regression/build_report.py` | Small wrapper that calls `report/page.py`. |
| `regression/baseline.py` | Shared split, scaler pipeline, metrics, and coefficient helpers. |
| `regression/legacy_utils.py` | Legacy lightweight regression utilities. |

### `regression/diagnostics/`

| File | Purpose |
| --- | --- |
| `correlation.py` | Standalone correlation diagnostic. |
| `multicollinearity.py` | Standalone VIF/correlation diagnostic. |
| `normalization.py` | Standalone normalized linear regression diagnostic. |

### `regression/report/`

| File | Purpose |
| --- | --- |
| `config.py` | Data paths, targets, feature lists, tree/XGBoost settings. |
| `page.py` | Builds `site/regression_analysis.html` and regression CSV outputs. |
| `modeling.py` | Main skew-aware linear regression baseline. |
| `skew_transform.py` | Train-fit skew transforms applied to train/test data. |
| `tree_models.py` | Decision tree, XGBoost, adoption tree, adoption XGBoost. |
| `ensemble.py` | Stacked ensemble with out-of-fold base predictions. |
| `lag_analysis.py` | Lagged revenue-impact analysis. |
| `nonlinearity.py` | Nonlinear correlation checks. |
| `profiles.py` | Variable profile and skewness helpers. |
| `tables.py` | CSV/table formatting helpers. |
| `visuals.py` | SVG/plot helpers for report visuals. |
| `formatting.py` | Shared number, money, percentage, and HTML formatting. |

### `charts/exploratory/`

These are exploratory chart scripts. The generated dashboard now handles most production views.

| File | Purpose |
| --- | --- |
| `adoption_rates.py` | Average AI adoption over time. |
| `automation_productivity.py` | Automation vs productivity scatter. |
| `adoption_productivity_correlation.py` | Adoption/productivity correlation view. |
| `deployment_to_training.py` | Deployment by training quartile. |
| `industry_metrics.py` | Net value by industry. |
| `investment_benefit_trend.py` | Investment vs benefit trend. |
| `investment_benefit_outliers.py` | Investment vs total benefit scatter. |
| `roi_quantiles.py` | ROI by investment quantile. |
| `training_quantiles.py` | Productivity/ROI by training quantile. |

### `charts/pages/`

| File | Purpose |
| --- | --- |
| `adoption_comparison.py` | Combined country/industry adoption comparison HTML generator. |

### `data/`

| File | Purpose |
| --- | --- |
| `dataset.py` | One shared dataset loader for app and charts. |
| `coverage.py` | Dataset coverage summary script. |

## Dataset

Original source:

```text
hassangasem/corporate-ai-adoption-and-roi-dataset-20152035
```

Active local copy:

```text
site/downloads/dashboard/full_dataset.csv
```

Shape:

```text
Rows: 200,000
Columns: 18
Years: 2015-2035
Countries: 15
Industries: 10
```

Columns:

```text
company_id, industry, country, year, ai_adoption_level,
ai_investment_usd, automation_rate, cost_savings, revenue_impact,
productivity_gain, employee_ai_training_hours, ai_maturity_score,
deployment_count, total_benefit, net_value, roi,
training_group, invest_group
```

## Generated Outputs

### HTML

```text
site/dashboard.html
site/industry_country_profiles.html
site/regression_analysis.html
site/country_adoption_rate.html
site/industry_adoption_rate.html
```

### Dashboard CSVs

```text
site/downloads/dashboard/full_dataset.csv
site/downloads/dashboard/summary.csv
site/downloads/dashboard/yearly_adoption.csv
site/downloads/dashboard/industry_adoption.csv
site/downloads/dashboard/country_adoption.csv
site/downloads/dashboard/adoption_productivity_by_year.csv
site/downloads/dashboard/industry_ai_profiles.csv
site/downloads/dashboard/country_ai_profiles.csv
```

### Regression CSVs

```text
site/downloads/regression/regression_statistics.csv
site/downloads/regression/regression_coefficients.csv
site/downloads/regression/regression_tree_model_selection.csv
site/downloads/regression/regression_xgboost_statistics.csv
site/downloads/regression/regression_xgboost_importance.csv
site/downloads/regression/regression_adoption_tree_statistics.csv
site/downloads/regression/regression_adoption_xgboost_statistics.csv
site/downloads/regression/regression_stacked_ensemble_statistics.csv
site/downloads/regression/regression_stacked_ensemble_coefficients.csv
site/downloads/regression/regression_largest_error_cases.csv
```

## Model Summary

Revenue target:

```text
revenue_impact
```

Revenue features:

```text
ai_investment_usd
automation_rate
cost_savings
employee_ai_training_hours
deployment_count
```

Adoption target:

```text
ai_adoption_level
```

Current interpretation:

```text
AI adoption is highly predictable from AI maturity and usage variables.
Revenue impact is only moderately predictable from the selected AI variables.
```

## Regeneration

Run:

```bash
.venv/bin/python aicoveragedata/app/dashboard.py
```

Serve:

```bash
.venv/bin/python aicoveragedata/app/dashboard_server.py
```

## Cleanup Rules

Keep:

```text
source code -> agent/, app/, data/, charts/, regression/
generated files -> site/
runtime memory -> agent/state/
```

If dataset paths move, update together:

```text
data/dataset.py
regression/report/config.py
agent/context/data_catalog.py
regression/legacy_utils.py
```
