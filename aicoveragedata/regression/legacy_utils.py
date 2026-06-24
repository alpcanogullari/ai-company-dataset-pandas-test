import csv
import html
import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SITE_DIR = PACKAGE_DIR / "site"
DATA_PATH = SITE_DIR / "downloads" / "dashboard" / "full_dataset.csv"
DOWNLOAD_DIR = SITE_DIR / "downloads" / "regression"
REGRESSION_PAGE = SITE_DIR / "regression_analysis.html"

TARGET_COLUMN = "revenue_impact"
FEATURE_COLUMNS = [
    "ai_investment_usd",
    "automation_rate",
    "cost_savings",
    "employee_ai_training_hours",
    "deployment_count",
]

PREFERRED_TARGET_COLUMNS = ["revenue_impact", "roi", "target", "outcome", "label", "y", "sales", "value"]
DEFAULT_FEATURE_COLUMNS = FEATURE_COLUMNS
ID_HINTS = ["id", "uuid", "key", "year", "record_index"]


def is_id_like(column):
    name = column.lower()
    return any(name == hint or name.startswith(f"{hint}_") or name.endswith(f"_{hint}") for hint in ID_HINTS)


def to_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def inspect_numeric_columns(data_path=DATA_PATH, sample_size=5000):
    counts = {}
    unique_values = {}
    with open(data_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        for column in fieldnames:
            counts[column] = 0
            unique_values[column] = set()

        for index, row in enumerate(reader):
            if index >= sample_size:
                break
            for column in fieldnames:
                value = to_float(row.get(column))
                if value is not None:
                    counts[column] += 1
                    if len(unique_values[column]) < 3:
                        unique_values[column].add(value)

    return [
        column
        for column in counts
        if counts[column] > 0 and len(unique_values[column]) > 1
    ]


def infer_target_and_features(data_path=DATA_PATH, target=None, features=None):
    numeric_columns = inspect_numeric_columns(data_path)
    if len(numeric_columns) < 2:
        raise ValueError("Regression needs at least two numeric columns.")

    target = target or next(
        (column for column in PREFERRED_TARGET_COLUMNS if column in numeric_columns),
        None,
    )
    if target is None:
        target = next((column for column in reversed(numeric_columns) if not is_id_like(column)), numeric_columns[-1])

    if features:
        selected_features = [column for column in features if column in numeric_columns and column != target]
    else:
        selected_features = [
            column
            for column in DEFAULT_FEATURE_COLUMNS
            if column in numeric_columns and column != target
        ]
        if not selected_features:
            selected_features = [
                column
                for column in numeric_columns
                if column != target and not is_id_like(column)
            ]

    if not selected_features:
        raise ValueError("Choose at least one numeric feature.")

    return target, selected_features, numeric_columns


def load_model_rows(data_path=DATA_PATH, target=None, features=None):
    target, features, numeric_columns = infer_target_and_features(data_path, target, features)
    rows = []
    with open(data_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            y = to_float(raw_row.get(target))
            x = [to_float(raw_row.get(feature)) for feature in features]
            if y is None or any(value is None for value in x):
                continue
            rows.append((x, y))

    if len(rows) <= len(features) + 2:
        raise ValueError("Not enough complete rows for this regression.")

    return rows, target, features, numeric_columns


def solve_linear_system(matrix, vector):
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]

    for pivot_index in range(n):
        pivot_row = max(range(pivot_index, n), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("Regression matrix is singular. Choose fewer or less duplicated features.")
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot = augmented[pivot_index][pivot_index]
        for column in range(pivot_index, n + 1):
            augmented[pivot_index][column] /= pivot

        for row in range(n):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for column in range(pivot_index, n + 1):
                augmented[row][column] -= factor * augmented[pivot_index][column]

    return [augmented[row][n] for row in range(n)]


def fit_linear_regression(data_path=DATA_PATH, target=None, features=None):
    rows, target, features, numeric_columns = load_model_rows(data_path, target, features)
    split_index = max(len(features) + 2, int(len(rows) * 0.70))
    if split_index >= len(rows):
        split_index = len(rows) - 1

    train_rows = rows[:split_index]
    test_rows = rows[split_index:]
    feature_count = len(features)
    means = [
        sum(row[0][feature_index] for row in train_rows) / len(train_rows)
        for feature_index in range(feature_count)
    ]
    stds = []
    for feature_index in range(feature_count):
        variance = sum((row[0][feature_index] - means[feature_index]) ** 2 for row in train_rows) / len(train_rows)
        stds.append(variance ** 0.5 or 1.0)

    size = feature_count + 1
    xtx = [[0.0 for _ in range(size)] for _ in range(size)]
    xty = [0.0 for _ in range(size)]

    for x_values, y_value in train_rows:
        scaled = [1.0] + [
            (x_values[index] - means[index]) / stds[index]
            for index in range(feature_count)
        ]
        for row_index in range(size):
            xty[row_index] += scaled[row_index] * y_value
            for column_index in range(size):
                xtx[row_index][column_index] += scaled[row_index] * scaled[column_index]

    for index in range(1, size):
        xtx[index][index] += 1e-8

    coefficients = solve_linear_system(xtx, xty)
    predictions = []
    errors = []
    actual_values = []
    for x_values, y_value in test_rows:
        scaled = [1.0] + [
            (x_values[index] - means[index]) / stds[index]
            for index in range(feature_count)
        ]
        predicted = sum(coefficients[index] * scaled[index] for index in range(size))
        error = y_value - predicted
        actual_values.append(y_value)
        errors.append(error)
        predictions.append(
            {
                "actual": y_value,
                "predicted": predicted,
                "error": error,
                "absolute_error": abs(error),
                "error_direction": "Underpredicted" if error > 0 else "Overpredicted" if error < 0 else "Exact",
            }
        )

    mean_actual = sum(actual_values) / len(actual_values)
    ss_res = sum(error * error for error in errors)
    ss_tot = sum((value - mean_actual) ** 2 for value in actual_values)
    r2 = 1 - (ss_res / ss_tot) if ss_tot else 0.0
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = (ss_res / len(errors)) ** 0.5

    coefficient_rows = [{"variable": "intercept", "standardized_coefficient": coefficients[0]}]
    coefficient_rows.extend(
        {
            "variable": feature,
            "standardized_coefficient": coefficients[index + 1],
        }
        for index, feature in enumerate(features)
    )
    coefficient_rows[1:] = sorted(
        coefficient_rows[1:],
        key=lambda row: abs(row["standardized_coefficient"]),
        reverse=True,
    )

    return {
        "target": target,
        "features": features,
        "numeric_columns": numeric_columns,
        "rows_used": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "coefficients": coefficient_rows,
        "predictions": predictions,
        "largest_errors": sorted(predictions, key=lambda row: row["absolute_error"], reverse=True)[:20],
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value):
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1_000_000:
            return f"{value:,.0f}"
        return f"{value:,.4f}"
    return str(value)


def table_html(rows, limit=30):
    if not rows:
        return "<p>No rows available.</p>"
    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        cells = "".join(
            f"<td>{html.escape(format_number(row.get(column, '')))}</td>"
            for column in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def export_regression_outputs(result):
    write_csv(DOWNLOAD_DIR / "regression_coefficients.csv", result["coefficients"])
    write_csv(DOWNLOAD_DIR / "regression_predictions_sample.csv", result["predictions"][:500])
    write_csv(DOWNLOAD_DIR / "selected_regression_largest_errors.csv", result["largest_errors"])


def regression_response(target=None, features=None):
    result = fit_linear_regression(target=target, features=features)
    export_regression_outputs(result)
    return {
        "metrics": {
            "Target": result["target"],
            "Features": len(result["features"]),
            "Rows Used": result["rows_used"],
            "Train Rows": result["train_rows"],
            "Test Rows": result["test_rows"],
            "R Squared": result["metrics"]["r2"],
            "MAE": result["metrics"]["mae"],
            "RMSE": result["metrics"]["rmse"],
        },
        "coefficients": result["coefficients"],
        "prediction_sample": result["predictions"][:500],
        "largest_errors": result["largest_errors"],
        "numeric_columns": result["numeric_columns"],
        "default_features": result["features"],
    }


def write_regression_page(output_path=REGRESSION_PAGE):
    response = regression_response()
    cards = "".join(
        f"<div class='stat'><span>{html.escape(label)}</span><strong>{html.escape(format_number(value))}</strong></div>"
        for label, value in response["metrics"].items()
    )
    features = ", ".join(response["default_features"])
    payload = html.escape(json.dumps(response))
    page = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Regression Analysis</title>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #f4f6f8; color: #17202a; }}
        header {{ padding: 28px 32px 18px; background: #ffffff; border-bottom: 1px solid #d8dee4; }}
        h1 {{ margin: 0 0 8px; font-size: 32px; }}
        h2 {{ margin: 0 0 12px; font-size: 20px; }}
        p {{ margin: 0; color: #5b6670; line-height: 1.45; }}
        a {{ color: #1769aa; text-decoration: none; }}
        .top-links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
        .top-links a, .download {{ display: inline-block; padding: 9px 12px; border: 1px solid #b7c2cc; border-radius: 6px; background: #ffffff; color: #17202a; font-size: 14px; }}
        .page {{ display: grid; gap: 18px; padding: 20px 32px 32px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
        .panel, .stat {{ background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; padding: 16px; }}
        .stat span {{ display: block; color: #5b6670; font-size: 13px; margin-bottom: 6px; }}
        .stat strong {{ font-size: 22px; word-break: break-word; }}
        .table-wrap {{ overflow-x: auto; border: 1px solid #d8dee4; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }}
        th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px 10px; text-align: right; }}
        th, td:first-child {{ text-align: left; }}
        th {{ background: #eef3f7; }}
        tr:nth-child(even) td {{ background: #fbfcfd; }}
        .download-list {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        code {{ background: #eef3f7; padding: 2px 5px; border-radius: 4px; }}
        @media (max-width: 520px) {{ header, .page {{ padding-left: 14px; padding-right: 14px; }} h1 {{ font-size: 25px; }} }}
    </style>
</head>
<body>
    <header>
        <h1>Regression Analysis</h1>
        <p>Predicts <code>{html.escape(response["metrics"]["Target"])}</code> from AI adoption drivers.</p>
        <div class="top-links">
            <a href="dashboard.html">Dashboard</a>
            <a href="industry_country_profiles.html">Industry and Country Profiles</a>
        </div>
    </header>
    <main class="page">
        <section class="grid">{cards}</section>
        <section class="panel">
            <h2>Model Setup</h2>
            <p>Features: {html.escape(features)}</p>
        </section>
        <section class="panel">
            <h2>Downloads</h2>
            <div class="download-list">
                <a class="download" href="downloads/regression/regression_coefficients.csv" download>Coefficients</a>
                <a class="download" href="downloads/regression/regression_predictions_sample.csv" download>Prediction Sample</a>
                <a class="download" href="downloads/regression/selected_regression_largest_errors.csv" download>Largest Errors</a>
            </div>
        </section>
        <section class="panel">
            <h2>Standardized Coefficients</h2>
            {table_html(response["coefficients"], limit=50)}
        </section>
        <section class="panel">
            <h2>Largest Prediction Errors</h2>
            {table_html(response["largest_errors"], limit=20)}
        </section>
    </main>
    <script type="application/json" id="regression-payload">{payload}</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def load_regression_data(data_path=DATA_PATH):
    rows, target, features, _ = load_model_rows(data_path)
    try:
        import pandas as pd
    except ModuleNotFoundError:
        X = [row[0] for row in rows]
        y = [row[1] for row in rows]
        return X, y, target, features

    X = pd.DataFrame([row[0] for row in rows], columns=features)
    y = pd.Series([row[1] for row in rows], name=target)
    return X, y, target, features


def print_dataset_summary(target, features, data_path=DATA_PATH):
    print("Dataset:", data_path)
    print("Target:", target)
    print("Features:", ", ".join(features))
