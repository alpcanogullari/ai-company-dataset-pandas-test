from functools import lru_cache
from pathlib import Path
import re

import pandas as pd

from .data_catalog import (
    DASHBOARD_FILES,
    FULL_DATASET,
    PACKAGE_DIR,
    PROJECT_EXCLUDE_PARTS,
    PROJECT_TEXT_SUFFIXES,
    REGRESSION_FILES,
    SITE_DIR,
)


MONEY_COLUMNS = {"ai_investment_usd", "cost_savings", "revenue_impact", "total_benefit", "net_value"}
RATE_COLUMNS = {"ai_adoption_level", "automation_rate", "productivity_gain", "roi"}
GROUP_COLUMNS = {"industry", "country", "year", "training_group", "invest_group"}
LOWEST_WORDS = {"bottom", "least", "lowest", "minimum", "smallest", "weakest", "worst"}
HIGHEST_WORDS = {"best", "greatest", "highest", "largest", "maximum", "most", "strongest", "top"}
GROUP_ALIASES = {
    "country": {"countries", "country", "nation", "nations"},
    "industry": {"industries", "industry", "sector", "sectors"},
    "year": {"time", "trend", "year", "years"},
}
METRIC_ALIASES = {
    "adoption": "ai_adoption_level",
    "adoption rate": "ai_adoption_level",
    "ai adoption": "ai_adoption_level",
    "ai adoption rate": "ai_adoption_level",
    "ai_adoption_rate": "ai_adoption_level",
    "automation": "automation_rate",
    "benefit": "total_benefit",
    "cost": "cost_savings",
    "deployment": "deployment_count",
    "impact": "revenue_impact",
    "investment": "ai_investment_usd",
    "maturity": "ai_maturity_score",
    "net value": "net_value",
    "profit": "net_value",
    "productivity": "productivity_gain",
    "return": "roi",
    "return on investment": "roi",
    "returns": "roi",
    "revenue": "revenue_impact",
    "roi": "roi",
    "saving": "cost_savings",
    "training": "employee_ai_training_hours",
}
COLUMN_ALIASES = {
    "ai_adoption_rate": "ai_adoption_level",
    "adoption_rate": "ai_adoption_level",
}
CODE_CONTEXT_WORDS = {
    "agent",
    "api",
    "app",
    "bug",
    "button",
    "buttons",
    "class",
    "chat",
    "chatbot",
    "clear",
    "code",
    "codebase",
    "css",
    "endpoint",
    "function",
    "handler",
    "html",
    "implemented",
    "implementation",
    "javascript",
    "page",
    "python",
    "route",
    "script",
    "server",
    "site",
    "source",
    "style",
    "tab",
    "tabs",
    "theme",
    "ui",
    "widget",
}
SEARCH_TERM_ALIASES = {
    "chatbot": ["chat", "agent"],
    "implemented": ["implementation", "def", "function"],
    "button": ["button", "btn"],
    "buttons": ["button", "btn"],
}
HTML_CONTEXT_WORDS = {"html", "page", "site", "ui", "style", "css", "theme"}
STOP_WORDS = {
    "about",
    "able",
    "also",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "our",
    "read",
    "that",
    "the",
    "this",
    "through",
    "what",
    "when",
    "where",
    "with",
}
SCHEMA_WORDS = {
    "column",
    "columns",
    "datatype",
    "dtypes",
    "feature",
    "features",
    "field",
    "fields",
    "input",
    "inputs",
    "predictor",
    "predictors",
    "schema",
    "target",
    "type",
    "types",
    "variable",
    "variables",
}
REGRESSION_WORDS = {"coefficient", "error", "model", "prediction", "predict", "regression", "r square", "r squared", "tree", "xgboost"}
CORRELATION_WORDS = {"affect", "correlation", "driver", "important", "influence", "relationship"}
OUTPUT_WORDS = {"download", "file", "output", "where"}
FOLLOW_UP_WORDS = {"also", "and", "compare", "else", "highest", "lowest", "same", "that", "them", "those"}
TASK_DATASET_RANKING = "dataset_ranking"
TASK_DATASET_SCHEMA = "dataset_schema"
TASK_REGRESSION_MODEL = "regression_model"
TASK_CODEBASE_HTML = "codebase_html"
TASK_GENERAL = "general"


@lru_cache(maxsize=1)
def load_dataset():
    if not FULL_DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {FULL_DATASET}")
    return pd.read_csv(FULL_DATASET)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def project_text_files():
    files = []
    for path in PACKAGE_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in PROJECT_TEXT_SUFFIXES:
            continue
        if PROJECT_EXCLUDE_PARTS.intersection(path.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: str(item.relative_to(PACKAGE_DIR.parent)))


def relative_path(path):
    return str(Path(path).relative_to(PACKAGE_DIR.parent))


def read_text_file(path, max_chars=120000):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    text = re.sub(r"data:image/[^\"')\s>]+", "[embedded image data omitted]", text)
    if len(text) > max_chars:
        return text[:max_chars] + "\n[File preview trimmed.]"
    return text


def search_terms(question):
    terms = []
    for term in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", question.lower()):
        clean = term.strip("./")
        if clean and clean not in STOP_WORDS and clean not in terms:
            terms.append(clean)
            for alias in SEARCH_TERM_ALIASES.get(clean, []):
                if alias not in STOP_WORDS and alias not in terms:
                    terms.append(alias)
    return terms[:16]


def wants_code_context(question):
    lowered = question.lower()
    if re.search(r"\b[\w/-]+\.(py|html|md)\b", lowered):
        return True
    return any(word in lowered for word in CODE_CONTEXT_WORDS)


def wants_html_context(question):
    lowered = question.lower()
    if re.search(r"\b[\w/-]+\.html\b", lowered):
        return True
    return any(word in lowered for word in HTML_CONTEXT_WORDS)


def file_overview():
    files = project_text_files()
    code_files = [path for path in files if path.suffix == ".py"]
    html_files = [path for path in files if path.suffix == ".html"]
    docs = [path for path in files if path.suffix == ".md"]
    lines = [
        "Project text file overview",
        f"- Python files: {len(code_files):,}",
        f"- HTML files: {len(html_files):,}",
        f"- Markdown files: {len(docs):,}",
    ]
    for label, group in [("Key Python files", code_files), ("Internal HTML files", html_files)]:
        lines.append(f"{label}:")
        for path in group[:24]:
            lines.append(f"- {relative_path(path)}")
    return "\n".join(lines)


def html_page_outline(path):
    text = read_text_file(path, max_chars=60000)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", text, flags=re.IGNORECASE | re.DOTALL)
    links = re.findall(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", text, flags=re.IGNORECASE | re.DOTALL)
    ids = re.findall(r"\bid=[\"']([^\"']+)[\"']", text)
    classes = re.findall(r"\bclass=[\"']([^\"']+)[\"']", text)

    def clean_html(value):
        value = re.sub(r"<[^>]+>", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    class_names = []
    for value in classes:
        for name in value.split():
            if name not in class_names:
                class_names.append(name)

    lines = [f"HTML page outline: {relative_path(path)}"]
    if title_match:
        lines.append(f"- Title: {clean_html(title_match.group(1))}")
    if headings:
        lines.append("- Headings: " + "; ".join(clean_html(item) for item in headings[:12]))
    if links:
        link_text = [f"{clean_html(label)} -> {href}" for href, label in links[:12]]
        lines.append("- Links: " + "; ".join(link_text))
    if ids:
        lines.append("- IDs: " + ", ".join(ids[:24]))
    if class_names:
        lines.append("- Classes: " + ", ".join(class_names[:32]))
    return "\n".join(lines)


def html_overview():
    html_files = [path for path in project_text_files() if path.suffix == ".html" and SITE_DIR in path.parents]
    if not html_files:
        return "No internal HTML files found."
    return "\n\n".join(html_page_outline(path) for path in html_files[:8])


def score_file_for_terms(path, terms):
    path_text = relative_path(path).lower()
    score = 0
    for term in terms:
        if term in path_text:
            score += 8
    text = read_text_file(path, max_chars=80000).lower()
    for term in terms:
        score += min(text.count(term), 8)
    return score


def snippets_for_file(path, terms, max_snippets=2, context_lines=2):
    text = read_text_file(path)
    if not text:
        return ""
    lines = text.splitlines()
    lowered_terms = [term.lower() for term in terms]
    indexes = []
    min_gap = (context_lines * 2) + 2
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(term in lowered for term in lowered_terms) and all(
            abs(index - existing) >= min_gap for existing in indexes
        ):
            indexes.append(index)
        if len(indexes) >= max_snippets:
            break
    if not indexes:
        indexes = [0]

    chunks = []
    used = set()
    for index in indexes:
        start = max(0, index - context_lines)
        end = min(len(lines), index + context_lines + 1)
        numbered = []
        for line_number in range(start, end):
            if line_number in used:
                continue
            used.add(line_number)
            content = lines[line_number]
            if len(content) > 220:
                content = content[:220] + " ..."
            numbered.append(f"{line_number + 1}: {content}")
        if numbered:
            chunks.append("\n".join(numbered))
    return "\n...\n".join(chunks)


def file_preview(path, max_lines=130):
    text = read_text_file(path)
    if not text:
        return ""
    numbered = []
    for index, line in enumerate(text.splitlines()[:max_lines]):
        if len(line) > 220:
            line = line[:220] + " ..."
        numbered.append(f"{index + 1}: {line}")
    return "\n".join(numbered)


def explicitly_mentioned_files(question, files):
    lowered = question.lower()
    matches = set()
    for path in files:
        rel = relative_path(path).lower()
        name = path.name.lower()
        if name in lowered or rel in lowered:
            matches.add(path)
    return matches


def project_search_context(question):
    terms = search_terms(question)
    if not terms:
        terms = ["dashboard", "agent", "server", "html"]

    files = project_text_files()
    explicit_files = explicitly_mentioned_files(question, files)
    scored = [
        (score_file_for_terms(path, terms), path)
        for path in files
    ]
    matches = [(score, path) for score, path in scored if score > 0]
    matches.sort(key=lambda item: (-item[0], relative_path(item[1])))

    sections = [
        "Internal code search context. Use these snippets as evidence, but do not mention searched files or paths unless the user asks where code lives.",
        file_overview(),
    ]
    if wants_html_context(question):
        sections.append(html_overview())

    if matches:
        snippet_lines = ["Relevant code and HTML snippets"]
        for score, path in matches[:8]:
            snippet = file_preview(path) if path in explicit_files else snippets_for_file(path, terms)
            snippet_lines.append(f"\nFile: {relative_path(path)} (score {score})\n{snippet}")
        sections.append("\n".join(snippet_lines))
    else:
        sections.append("No matching code or HTML snippets found for this question.")
    return "\n\n---\n\n".join(sections)


def format_value(value, column=None):
    if pd.isna(value):
        return "n/a"
    if column in MONEY_COLUMNS:
        return f"${float(value):,.0f}"
    if column in RATE_COLUMNS:
        return f"{float(value):,.4f}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def frame_preview(frame, rows=8, cols=8):
    if frame is None or frame.empty:
        return "No data available."
    small = frame.head(rows).iloc[:, :cols].copy()
    return small.to_string(index=False)


def dataset_overview():
    df = load_dataset()
    numeric = df.select_dtypes(include="number")
    categorical = [column for column in df.columns if column not in numeric.columns]
    lines = [
        "Dataset overview",
        f"- Source: {FULL_DATASET}",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns):,}",
        f"- Year range: {int(df['year'].min())} to {int(df['year'].max())}" if "year" in df else "- Year range: n/a",
        f"- Industries: {df['industry'].nunique():,}" if "industry" in df else "- Industries: n/a",
        f"- Countries: {df['country'].nunique():,}" if "country" in df else "- Countries: n/a",
        f"- Numeric columns: {', '.join(numeric.columns)}",
        f"- Categorical columns: {', '.join(categorical)}",
    ]
    missing = df.isna().sum()
    missing = missing[missing > 0]
    lines.append("- Missing values: none" if missing.empty else "- Missing values:\n" + missing.to_string())
    return "\n".join(lines)


def list_columns():
    df = load_dataset()
    details = []
    for column in df.columns:
        details.append(f"- {column}: {df[column].dtype}, unique={df[column].nunique(dropna=True):,}")
    return "Available dataset columns\n" + "\n".join(details)


def column_profile(column):
    df = load_dataset()
    if column not in df.columns:
        return f"Column not found: {column}"

    series = df[column]
    lines = [f"Column profile: {column}", f"- Type: {series.dtype}", f"- Missing: {series.isna().sum():,}"]
    if pd.api.types.is_numeric_dtype(series):
        stats = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        for label in ["min", "10%", "25%", "50%", "mean", "75%", "90%", "max"]:
            if label in stats:
                lines.append(f"- {label}: {format_value(stats[label], column)}")
    else:
        top = series.value_counts(dropna=False).head(10)
        lines.append("- Top values:")
        for value, count in top.items():
            lines.append(f"  - {value}: {count:,}")
    return "\n".join(lines)


def compare_by_group(metric, group_col, top_n=10, ascending=False):
    df = load_dataset()
    if metric not in df.columns:
        return f"Metric not found: {metric}"
    if group_col not in df.columns:
        return f"Group column not found: {group_col}"
    if not pd.api.types.is_numeric_dtype(df[metric]):
        return f"Metric is not numeric: {metric}"

    grouped = (
        df.groupby(group_col, observed=True)
        .agg(mean_value=(metric, "mean"), median_value=(metric, "median"), rows=(metric, "size"))
        .reset_index()
        .sort_values("mean_value", ascending=ascending)
        .head(top_n)
    )
    grouped["mean_value"] = grouped["mean_value"].map(lambda value: format_value(value, metric))
    grouped["median_value"] = grouped["median_value"].map(lambda value: format_value(value, metric))
    grouped["rows"] = grouped["rows"].map(lambda value: f"{int(value):,}")
    label = "Lowest" if ascending else "Top"
    return f"{label} {top_n} {group_col} values by average {metric}\n{frame_preview(grouped, rows=top_n, cols=4)}"


def ranking_evidence(metric, group_col, direction):
    requested_lowest = direction == "lowest"
    requested = compare_by_group(metric, group_col, ascending=requested_lowest)
    opposite = compare_by_group(metric, group_col, ascending=not requested_lowest)
    opposite_label = "highest" if requested_lowest else "lowest"
    return (
        f"{requested}\n\n"
        f"Opposite-side evidence for follow-up questions ({opposite_label}):\n"
        f"{opposite}"
    )


def contains_any_word(text, words):
    return any(re.search(rf"\b{word}\b", text) for word in words)


def contains_any_phrase(text, phrases):
    return any(phrase in text for phrase in phrases)


def ranking_direction(question, default="highest"):
    lowered = question.lower()
    if contains_any_word(lowered, LOWEST_WORDS):
        return "lowest"
    if contains_any_word(lowered, HIGHEST_WORDS):
        return "highest"
    return default


def recent_user_context(history_messages, limit=3):
    if not history_messages:
        return ""
    user_messages = [
        str(message.get("content", "")).strip()
        for message in history_messages
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    return "\n".join(user_messages[-limit:])


def recent_user_messages(history_messages, limit=6):
    if not history_messages:
        return []
    messages = [
        str(message.get("content", "")).strip()
        for message in history_messages
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    return messages[-limit:]


def inference_question(question, history_messages=None):
    prior = recent_user_context(history_messages)
    if not prior:
        return question
    return f"{prior}\n{question}"


def expand_column_aliases(text):
    expanded = text
    lowered = text.lower()
    for alias, column in COLUMN_ALIASES.items():
        if alias in lowered and column not in lowered:
            expanded = f"{expanded}\n{alias} refers to dataset column {column}."
    return expanded


def detect_group_columns(question):
    lowered = question.lower()
    groups = []
    for group_col, aliases in GROUP_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
            groups.append(group_col)
    return groups


def find_metric(text):
    df = load_dataset()
    lowered = text.lower()
    for word, column in METRIC_ALIASES.items():
        if word in lowered and column in df.columns:
            return column
    for column in detect_columns(text):
        if pd.api.types.is_numeric_dtype(df[column]):
            return column
    return None


def intent_from_text(text):
    expanded = expand_column_aliases(text)
    return {
        "metric": find_metric(expanded),
        "groups": detect_group_columns(expanded),
        "direction": ranking_direction(expanded, default=None),
        "has_ranking": contains_any_word(expanded.lower(), LOWEST_WORDS | HIGHEST_WORDS),
    }


def looks_like_followup(question):
    lowered = question.lower()
    if len(search_terms(question)) <= 4 and any(word in lowered for word in FOLLOW_UP_WORDS):
        return True
    return lowered.startswith(("what about", "how about", "and ", "same for", "compare to"))


def resolve_dataset_intent(question, history_messages=None):
    current = intent_from_text(question)
    prior_intents = [intent_from_text(message) for message in recent_user_messages(history_messages)]
    prior_rankings = [intent for intent in prior_intents if intent["has_ranking"] or intent["groups"]]

    prior_metric = next((intent["metric"] for intent in reversed(prior_intents) if intent["metric"]), None)
    prior_groups = next((intent["groups"] for intent in reversed(prior_intents) if intent["groups"]), [])
    prior_direction = next((intent["direction"] for intent in reversed(prior_rankings) if intent["direction"]), None)

    metric = current["metric"] or prior_metric or "revenue_impact"
    groups = current["groups"] or prior_groups
    direction = current["direction"] or prior_direction or "highest"
    ranking = bool(current["has_ranking"] or groups or (looks_like_followup(question) and prior_rankings))

    return {
        "metric": metric,
        "groups": groups,
        "direction": direction,
        "ranking": ranking,
        "used_history": bool((not current["metric"] and prior_metric) or (not current["groups"] and prior_groups) or (not current["direction"] and prior_direction)),
    }


def data_question_interpretation(metric, group_cols, direction):
    lines = [
        "Dataset question interpretation",
        f"- Metric inferred: {metric}",
        f"- Grouping inferred: {', '.join(group_cols) if group_cols else 'none'}",
        f"- Ranking direction: {direction}",
    ]
    if group_cols:
        order = "ascending" if direction == "lowest" else "descending"
        lines.append(f"- Ranking rule: group rows by the inferred field and sort average {metric} {order}.")
        lines.append("- Answer rule: use the first row of the matching ranked table.")
    return "\n".join(lines)


def followup_resolution(intent):
    if not intent["used_history"]:
        return ""
    lines = [
        "Follow-up resolution",
        f"- Metric used: {intent['metric']}",
        f"- Grouping used: {', '.join(intent['groups']) if intent['groups'] else 'none'}",
        f"- Direction used: {intent['direction']}",
        "- Missing parts were inherited from recent dataset questions.",
    ]
    return "\n".join(lines)


def trend_by_year(metric):
    df = load_dataset()
    if "year" not in df.columns:
        return "No year column is available."
    if metric not in df.columns:
        return f"Metric not found: {metric}"
    if not pd.api.types.is_numeric_dtype(df[metric]):
        return f"Metric is not numeric: {metric}"

    trend = df.groupby("year", observed=True)[metric].mean().reset_index()
    trend[metric] = trend[metric].map(lambda value: format_value(value, metric))
    return f"Yearly average for {metric}\n{frame_preview(trend, rows=30, cols=2)}"


def correlations_for(target, top_n=10):
    df = load_dataset()
    if target not in df.columns:
        return f"Target column not found: {target}"
    if not pd.api.types.is_numeric_dtype(df[target]):
        return f"Target is not numeric: {target}"

    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)
    table = corr.reset_index()
    table.columns = ["variable", "pearson_correlation"]
    return f"Top correlations with {target}\n{frame_preview(table, rows=top_n, cols=2)}"


def regression_summary():
    sections = [
        "Regression and model outputs",
        "Target map: revenue models use revenue_impact. Supervised adoption tree and adoption XGBoost use ai_adoption_level. User phrasing like ai_adoption_rate refers to ai_adoption_level in this dataset.",
    ]
    for label, key, rows in [
        ("Skew transform audit", "skew_transform_audit", 12),
        ("Linear regression statistics", "statistics", 10),
        ("Linear coefficients", "coefficients", 12),
        ("Model selection", "model_selection", 10),
        ("Final selected model stats", "final_model", 12),
        ("XGBoost statistics", "xgboost_statistics", 12),
        ("XGBoost feature importance", "xgboost_importance", 10),
        ("Supervised adoption tree statistics", "adoption_tree_statistics", 12),
        ("Supervised adoption tree importance", "adoption_tree_importance", 10),
        ("Adoption XGBoost statistics", "adoption_xgboost_statistics", 12),
        ("Adoption XGBoost feature importance", "adoption_xgboost_importance", 10),
        ("Stacked ensemble regression statistics", "stacked_ensemble_statistics", 12),
        ("Stacked ensemble regression coefficients", "stacked_ensemble_coefficients", 8),
    ]:
        frame = read_csv(REGRESSION_FILES[key])
        sections.append(f"\n{label}\n{frame_preview(frame, rows=rows, cols=8)}")
    return "\n".join(sections)


def available_outputs():
    lines = ["Available generated outputs"]
    lines.append("Dashboard files:")
    for name, path in DASHBOARD_FILES.items():
        lines.append(f"- {name}: {path}")
    lines.append("Regression files:")
    for name, path in REGRESSION_FILES.items():
        lines.append(f"- {name}: {path}")
    return "\n".join(lines)


def detect_columns(question):
    df = load_dataset()
    lowered = question.lower()
    found = []
    for column in df.columns:
        variants = {column.lower(), column.replace("_", " ").lower()}
        if any(re.search(rf"\b{re.escape(variant)}\b", lowered) for variant in variants):
            found.append(column)
    return found


def guess_metric(question, fallback_question=None):
    for text in [question, fallback_question]:
        if not text:
            continue
        metric = find_metric(text)
        if metric:
            return metric
    return "revenue_impact"


def classify_context_task(question, history_messages=None):
    inference_text = expand_column_aliases(inference_question(question, history_messages))
    lowered = inference_text.lower()
    intent = resolve_dataset_intent(question, history_messages)

    if wants_code_context(inference_text) or wants_html_context(inference_text):
        return TASK_CODEBASE_HTML
    if intent["ranking"] or contains_any_word(lowered, LOWEST_WORDS | HIGHEST_WORDS):
        return TASK_DATASET_RANKING
    if contains_any_phrase(lowered, REGRESSION_WORDS):
        return TASK_REGRESSION_MODEL
    if contains_any_word(lowered, SCHEMA_WORDS | {"available", "components", "missing"}):
        return TASK_DATASET_SCHEMA
    return TASK_GENERAL


def task_label(task):
    labels = {
        TASK_DATASET_RANKING: "dataset ranking",
        TASK_DATASET_SCHEMA: "dataset schema",
        TASK_REGRESSION_MODEL: "regression/model analysis",
        TASK_CODEBASE_HTML: "codebase and internal HTML",
        TASK_GENERAL: "general dataset",
    }
    return labels.get(task, labels[TASK_GENERAL])


def collect_context(question, history_messages=None, task=None):
    lowered = question.lower()
    inference_text = expand_column_aliases(inference_question(question, history_messages))
    inference_lowered = inference_text.lower()
    task = task or classify_context_task(question, history_messages)
    intent = resolve_dataset_intent(question, history_messages)
    metric = intent["metric"]
    direction = intent["direction"]
    group_cols = intent["groups"]
    sections = [f"Question route: {task_label(task)}"]
    if task != TASK_CODEBASE_HTML:
        sections.append(dataset_overview())

    if task == TASK_DATASET_RANKING or group_cols or contains_any_word(inference_lowered, LOWEST_WORDS | HIGHEST_WORDS):
        resolution = followup_resolution(intent)
        if resolution:
            sections.append(resolution)
        sections.append(data_question_interpretation(metric, group_cols, direction))

    if task == TASK_DATASET_SCHEMA or contains_any_word(inference_lowered, SCHEMA_WORDS | {"available", "components", "missing"}):
        sections.append(list_columns())

    for column in detect_columns(inference_text)[:4]:
        sections.append(column_profile(column))

    if "industry" in group_cols:
        sections.append(ranking_evidence(metric, "industry", direction))
    if "country" in group_cols:
        sections.append(ranking_evidence(metric, "country", direction))
    if "year" in group_cols:
        if contains_any_word(lowered, LOWEST_WORDS | HIGHEST_WORDS):
            sections.append(ranking_evidence(metric, "year", direction))
        else:
            sections.append(trend_by_year(metric))

    if task == TASK_REGRESSION_MODEL or contains_any_word(lowered, CORRELATION_WORDS):
        sections.append(correlations_for(metric))

    if task == TASK_REGRESSION_MODEL:
        sections.append(regression_summary())

    if task != TASK_CODEBASE_HTML and contains_any_word(lowered, OUTPUT_WORDS):
        sections.append(available_outputs())

    if task == TASK_CODEBASE_HTML or wants_code_context(question):
        sections.append(project_search_context(question))

    return "\n\n---\n\n".join(sections)
