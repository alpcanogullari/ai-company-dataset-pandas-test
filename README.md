# AI Coverage Data Dashboard

Local dashboard, regression report, and DSPy-backed chatbot for the corporate AI adoption and ROI dataset.

## Run

Install dependencies:

```bash
.venv/bin/pip install -r aicoveragedata/agent/requirements.txt
```

Generate the site:

```bash
.venv/bin/python aicoveragedata/app/dashboard.py
```

Serve locally:

```bash
.venv/bin/python aicoveragedata/app/dashboard_server.py
```

Open:

```text
http://127.0.0.1:8000/
```

## Mental Map

```text
aicoveragedata/
  app/         builds and serves the website
  agent/       chatbot, memory, DSPy, and local context tools
  data/        dataset loading and coverage summary
  charts/      standalone chart scripts and generated chart-page builders
  regression/  regression models and report generation
  site/        generated HTML, CSV downloads, and reports
```

Source code lives outside `site/`.
Generated outputs live inside `site/`.

## Most Important Files

| File | What it does |
| --- | --- |
| `aicoveragedata/app/dashboard.py` | Main site generator. Builds dashboard, profile pages, and regression report. |
| `aicoveragedata/app/dashboard_server.py` | Local HTTP server. Serves `site/` and exposes `/api/agent`. |
| `aicoveragedata/app/agent_widget.py` | Floating chatbot tab, chat UI, history, clear button, and frontend API calls. |
| `aicoveragedata/data/dataset.py` | Single shared dataset loader. Reads local CSV first, Kaggle fallback second. |
| `aicoveragedata/data/coverage.py` | Dataset coverage summary script. |
| `aicoveragedata/charts/pages/adoption_comparison.py` | Combined country/industry adoption comparison page generator. |
| `aicoveragedata/regression/build_report.py` | Small entrypoint for the regression report builder. |
| `aicoveragedata/regression/baseline.py` | Shared baseline helpers: split, scaler pipeline, metrics, coefficients. |
| `aicoveragedata/regression/legacy_utils.py` | Legacy lightweight regression page/utilities kept for old diagnostics. |
| `aicoveragedata/regression/report/page.py` | Full regression report builder and CSV exporter. |
| `aicoveragedata/regression/report/config.py` | Regression targets, features, paths, and model settings. |
| `aicoveragedata/regression/report/modeling.py` | Skew-aware linear regression baseline. |
| `aicoveragedata/regression/report/tree_models.py` | Decision tree, XGBoost, adoption tree, adoption XGBoost. |
| `aicoveragedata/regression/report/ensemble.py` | Stacked ensemble using linear, tree, and XGBoost prediction signals. |
| `aicoveragedata/agent/core/agent.py` | Main chatbot answer flow. |
| `aicoveragedata/agent/context/tools.py` | Reads dataset, regression CSVs, code snippets, and HTML context. |
| `aicoveragedata/agent/llm/dspy_agent.py` | DSPy module connected to the local context tools. |
| `aicoveragedata/agent/memory/history.py` | Chat memory storage and formatting. |

## Dataset

Original source:

```text
hassangasem/corporate-ai-adoption-and-roi-dataset-20152035
```

Active local dataset:

```text
aicoveragedata/site/downloads/dashboard/full_dataset.csv
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

## Regression Models

Main revenue target:

```text
revenue_impact
```

Default revenue features:

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

Model outputs:

```text
aicoveragedata/site/downloads/regression/
```

Key outputs:

```text
regression_statistics.csv
regression_tree_model_selection.csv
regression_xgboost_statistics.csv
regression_adoption_tree_statistics.csv
regression_adoption_xgboost_statistics.csv
regression_stacked_ensemble_statistics.csv
```

Current high-level result:

```text
Revenue impact is moderately predictable.
AI adoption level is strongly predictable.
```

## Chatbot

Frontend:

```text
aicoveragedata/app/agent_widget.py
```

Backend:

```text
POST /api/agent
```

Agent flow:

```text
dashboard_server.py
-> agent/core/agent.py
-> agent/context/tools.py
-> agent/llm/dspy_agent.py
-> local dataset/regression/code context
```

Local memory:

```text
aicoveragedata/agent/state/chat_history.json
```

Local `.env`:

```text
aicoveragedata/agent/.env
```

Example:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5
OPENAI_REASONING_EFFORT=low
OPENAI_TEXT_VERBOSITY=low
AGENT_USE_DSPY=true
AGENT_DSPY_FALLBACK=true
```

## Generated Site

Generated HTML:

```text
aicoveragedata/site/index.html
aicoveragedata/site/industry_country_profiles.html
aicoveragedata/site/regression_analysis.html
```

Generated downloads:

```text
aicoveragedata/site/downloads/dashboard/
aicoveragedata/site/downloads/regression/
```

Generated reports:

```text
aicoveragedata/site/reports/
```

## Cleanup Notes

Recent placement cleanup:

```text
regression/evaluation.py
regression/interpretation.py
regression/model_pipeline.py
```

were merged into:

```text
regression/baseline.py
```

Moved folders:

```text
static/data_utils.py -> data/dataset.py
static/coverage_script.py -> data/coverage.py
static/regression_utils.py -> regression/legacy_utils.py
regression/*_script.py -> regression/diagnostics/
regression/regression_analysis_page.py -> regression/build_report.py
graphs/*.py -> charts/exploratory/
```

Merged duplicate chart page generators:

```text
graphs/country_adoption_rate.py
graphs/industry_adoption_rate.py
-> charts/pages/adoption_comparison.py
```

## Detailed Inventory

See:

```text
aicoveragedata/DATA_INVENTORY.md
```
