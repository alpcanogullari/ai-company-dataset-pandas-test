# AI Coverage Data Agent

Ask questions about the local dataset and regression outputs.

## Setup

Install the agent dependencies:

```bash
.venv/bin/pip install -r aicoveragedata/agent/requirements.txt
```

Edit the placeholder key file:

```bash
OPENAI_API_KEY=your_real_key_here
OPENAI_MODEL=your_openai_model
OPENAI_REASONING_EFFORT=low
OPENAI_TEXT_VERBOSITY=low
AGENT_USE_DSPY=true
AGENT_DSPY_FALLBACK=true
```

The real key goes in:

```text
aicoveragedata/agent/.env
```

That file is ignored by git.

`OPENAI_API_KEY` is the same key format for OpenAI API access. The model version is controlled by `OPENAI_MODEL`.

DSPy is enabled by default. The DSPy agent module uses the existing context builder and chat-history formatter directly, then sends grounded context to the model.

The DSPy flow uses routing, but not hardcoded answers:

```text
question -> route type -> gather matching evidence -> DSPy answers from evidence
```

Current route types:

- `dataset_ranking`: top, lowest, highest, country, industry, or year rankings.
- `regression_model`: regression, XGBoost, tree, R Square, coefficient, and prediction questions.
- `codebase_html`: local Python, dashboard, chatbot, CSS, and internal HTML questions.
- `dataset_schema`: variables, columns, fields, and feature questions.
- `general`: general dataset questions.

Example:

```text
"highest adoption?" after "lowest country adoption?"
```

The router keeps the topic as a dataset ranking question, gathers the country adoption evidence, and DSPy answers from the ranked table.

To bypass DSPy and use the direct OpenAI Responses API path:

```bash
AGENT_USE_DSPY=false
```

## Run

Single question:

```bash
.venv/bin/python -m aicoveragedata.agent.interface.cli -q "What affects revenue impact the most?"
```

Interactive mode:

```bash
.venv/bin/python -m aicoveragedata.agent.interface.cli
```

Offline local-data test:

```bash
.venv/bin/python -m aicoveragedata.agent.interface.cli --offline -q "Explain the XGBoost model"
```

Run local evaluation cases:

```bash
.venv/bin/python -m aicoveragedata.agent.evaluation.run_eval
```

Run live DSPy answer checks:

```bash
.venv/bin/python -m aicoveragedata.agent.evaluation.run_eval --live
```

## Folder Layout

```text
aicoveragedata/agent/
  core/       main answer flow and config
  context/    dataset, regression, code, and HTML context tools
  llm/        DSPy module and fallback prompt
  memory/     chat history storage
  interface/  CLI implementation
  evaluation/ reusable question cases and eval runner
  state/      ignored local runtime state
```

## Chat Memory

The frontend chat stores prior turns by browser session.
When the chat tab opens again, it reloads the stored messages for that browser session.
It also keeps a browser-side copy in `localStorage`, so switching between dashboard tabs/pages restores the visible chat immediately.

Stored locally:

```text
aicoveragedata/agent/state/chat_history.json
```

That file is ignored by git.

Use the chat panel's `Clear` button to reset the current browser session.

## What It Reads

Main dataset:

```text
aicoveragedata/site/downloads/dashboard/full_dataset.csv
```

Regression outputs:

```text
aicoveragedata/site/downloads/regression/*.csv
```

Examples:

- `regression_xgboost_statistics.csv`
- `regression_xgboost_importance.csv`
- `regression_tree_model_selection.csv`
- `regression_coefficients.csv`
