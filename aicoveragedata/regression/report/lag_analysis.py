import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import FEATURES, LAG_DECAY_FEATURES, TARGET
from .formatting import error_direction, money, scale_value, svg_text
from .modeling import build_metrics, metrics_table, pipeline_audit_table

def build_lagged_frame(model_df):
    lag_source_columns = [TARGET] + FEATURES
    lag_df = model_df[["company_id", "year"] + lag_source_columns].sort_values(
        ["company_id", "year"]
    ).copy()
    lag_df["Prior Year"] = lag_df.groupby("company_id")["year"].shift(1)

    lag_columns = []
    for column in lag_source_columns:
        lag_column = f"lag_1_{column}"
        lag_df[lag_column] = lag_df.groupby("company_id")[column].shift(1)
        lag_columns.append(lag_column)

    lag_df = lag_df[lag_df["Prior Year"] == lag_df["year"] - 1].dropna(
        subset=lag_columns + [TARGET]
    )
    return lag_df, lag_columns


def split_lagged_data_by_year(lag_df):
    train_index, test_index = train_test_split(
        lag_df.index,
        test_size=0.30,
        random_state=42,
    )
    train_mask = lag_df.index.isin(train_index)
    test_mask = lag_df.index.isin(test_index)
    train_years = sorted(lag_df.loc[train_mask, "year"].unique())
    test_years = sorted(lag_df.loc[test_mask, "year"].unique())
    return train_mask, test_mask, train_years, test_years


def fit_lagged_model(model_df):
    lag_df, lag_columns = build_lagged_frame(model_df)
    train_mask, test_mask, train_years, test_years = split_lagged_data_by_year(lag_df)

    X_train = lag_df.loc[train_mask, lag_columns]
    X_test = lag_df.loc[test_mask, lag_columns]
    y_train = lag_df.loc[train_mask, TARGET]
    y_test = lag_df.loc[test_mask, TARGET]

    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    metrics = build_metrics(y_train, train_pred, y_test, test_pred, len(lag_columns))

    coefficients = pd.DataFrame(
        {
            "Lagged Feature": lag_columns,
            "Original Variable": [column.replace("lag_1_", "") for column in lag_columns],
            "Coefficient": pipeline.named_steps["model"].coef_,
        }
    )
    coefficients["Abs Coefficient"] = coefficients["Coefficient"].abs()
    coefficients["Direction"] = np.where(coefficients["Coefficient"] >= 0, "Positive", "Negative")
    coefficients = coefficients.sort_values("Abs Coefficient", ascending=False)
    intercept = pd.DataFrame(
        [
            {
                "Lagged Feature": "Intercept",
                "Original Variable": "Intercept",
                "Coefficient": pipeline.named_steps["model"].intercept_,
                "Abs Coefficient": abs(pipeline.named_steps["model"].intercept_),
                "Direction": "Positive",
            }
        ]
    )
    coefficients = pd.concat([intercept, coefficients], ignore_index=True)

    predictions = lag_df.loc[
        test_mask,
        ["company_id", "year", "Prior Year"] + lag_columns,
    ].copy()
    predictions["Case ID"] = predictions.index
    predictions["Actual Revenue Impact"] = y_test
    predictions["Predicted Revenue Impact"] = test_pred
    predictions["Error"] = y_test - test_pred
    predictions["Error Direction"] = predictions["Error"].map(error_direction)
    predictions["Absolute Error"] = predictions["Error"].abs()
    predictions = predictions.sort_values("Case ID")
    largest = predictions.nlargest(5, "Absolute Error").drop(columns=["Absolute Error"])

    stats = metrics_table(metrics)
    stats.loc[len(stats)] = {
        "Metric": "Training Years Covered",
        "Training Data": f"{int(min(train_years))} - {int(max(train_years))}",
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Test Years Covered",
        "Training Data": "",
        "Test Data": f"{int(min(test_years))} - {int(max(test_years))}",
    }

    audit = pipeline_audit_table(
        pipeline,
        lag_columns,
        metrics,
        "random 70/30 split after lag rows, random_state=42",
    )

    return {
        "pipeline": pipeline,
        "lag_df": lag_df,
        "lag_columns": lag_columns,
        "metrics": metrics,
        "stats": stats,
        "audit": audit,
        "coefficients": coefficients,
        "predictions": predictions.drop(columns=["Absolute Error"]),
        "largest": largest,
    }


def fit_current_plus_lag_model(model_df):
    lag_df, lag_columns = build_lagged_frame(model_df)
    train_mask, test_mask, train_years, test_years = split_lagged_data_by_year(lag_df)
    feature_columns = FEATURES + lag_columns

    X_train = lag_df.loc[train_mask, feature_columns]
    X_test = lag_df.loc[test_mask, feature_columns]
    y_train = lag_df.loc[train_mask, TARGET]
    y_test = lag_df.loc[test_mask, TARGET]

    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    metrics = build_metrics(y_train, train_pred, y_test, test_pred, len(feature_columns))

    coefficient_lookup = pd.Series(
        pipeline.named_steps["model"].coef_,
        index=feature_columns,
    )
    coefficients = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Coefficient": pipeline.named_steps["model"].coef_,
        }
    )
    coefficients["Abs Coefficient"] = coefficients["Coefficient"].abs()
    coefficients["Direction"] = np.where(coefficients["Coefficient"] >= 0, "Positive", "Negative")
    coefficients = coefficients.sort_values("Abs Coefficient", ascending=False)

    component_rows = []
    for variable in [TARGET] + FEATURES:
        lag_feature = f"lag_1_{variable}"
        current_coefficient = coefficient_lookup.get(variable, np.nan)
        lag_coefficient = coefficient_lookup.get(lag_feature, np.nan)
        component_rows.append(
            {
                "Variable": variable,
                "Current Feature": variable if variable in FEATURES else "",
                "Current Coefficient": current_coefficient,
                "Lag Feature": lag_feature,
                "Lag Coefficient": lag_coefficient,
                "Lag Direction": "Positive" if lag_coefficient >= 0 else "Negative",
                "Abs Lag Coefficient": abs(lag_coefficient),
            }
        )
    components = pd.DataFrame(component_rows).sort_values("Abs Lag Coefficient", ascending=False)

    predictions = lag_df.loc[
        test_mask,
        ["company_id", "year", "Prior Year"] + feature_columns,
    ].copy()
    predictions["Case ID"] = predictions.index
    predictions["Actual Revenue Impact"] = y_test
    predictions["Predicted Revenue Impact"] = test_pred
    predictions["Error"] = y_test - test_pred
    predictions["Error Direction"] = predictions["Error"].map(error_direction)
    predictions["Absolute Error"] = predictions["Error"].abs()
    predictions = predictions.sort_values("Case ID")

    stats = metrics_table(metrics)
    stats.loc[len(stats)] = {
        "Metric": "Training Years Covered",
        "Training Data": f"{int(min(train_years))} - {int(max(train_years))}",
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Test Years Covered",
        "Training Data": "",
        "Test Data": f"{int(min(test_years))} - {int(max(test_years))}",
    }

    audit = pipeline_audit_table(
        pipeline,
        feature_columns,
        metrics,
        "random 70/30 split after lag rows, random_state=42",
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "stats": stats,
        "audit": audit,
        "coefficients": coefficients,
        "components": components,
        "predictions": predictions.drop(columns=["Absolute Error"]),
        "largest": predictions.nlargest(5, "Absolute Error").drop(columns=["Absolute Error"]),
    }


def build_time_lag_data(model_df):
    lag_df, lag_columns = build_lagged_frame(model_df)
    lag_df["Revenue Change From Prior Year"] = (
        lag_df[TARGET] - lag_df[f"lag_1_{TARGET}"]
    )

    overall_rows = []
    for column in lag_columns:
        overall_rows.append(
            {
                "Lagged Variable": column,
                "Current Target": TARGET,
                "Pearson Correlation": lag_df[column].corr(lag_df[TARGET]),
            }
        )

    yearly_rows = []
    for year, group in lag_df.groupby("year"):
        row = {
            "year": int(year),
            "Companies": group["company_id"].nunique(),
            "Current_Revenue_Impact": group[TARGET].mean(),
            "Prior_Year_Revenue_Impact": group[f"lag_1_{TARGET}"].mean(),
            "Revenue_Change_From_Prior_Year": group["Revenue Change From Prior Year"].mean(),
        }
        for column in lag_columns:
            row[f"{column}_Correlation_With_Current_Revenue"] = group[column].corr(group[TARGET])
        yearly_rows.append(row)

    yearly = pd.DataFrame(yearly_rows).sort_values("year")
    overall = pd.DataFrame(overall_rows).sort_values(
        "Pearson Correlation",
        key=abs,
        ascending=False,
    )
    summary = pd.DataFrame(
        [
            {
                "Metric": "Company-Year Prior Revenue Correlation",
                "Description": "Prior-year revenue impact vs current-year revenue impact across company-year rows",
                "Pearson Correlation": lag_df[f"lag_1_{TARGET}"].corr(lag_df[TARGET]),
            },
            {
                "Metric": "Year-Average Prior Revenue Correlation",
                "Description": "Average prior-year revenue impact vs average current-year revenue impact across years",
                "Pearson Correlation": yearly["Prior_Year_Revenue_Impact"].corr(yearly["Current_Revenue_Impact"]),
            },
        ]
    )
    return yearly, overall, summary


def render_time_lag_detail(yearly, overall, summary):
    chart_overall = overall.sort_values("Pearson Correlation")
    year_average_correlation = summary.loc[
        summary["Metric"] == "Year-Average Prior Revenue Correlation",
        "Pearson Correlation",
    ].iloc[0]

    bar_width = 760
    bar_height = 260
    left = 250
    right = 36
    top = 28
    row_height = 34
    zero_x = left + (bar_width - left - right) / 2
    bars = []
    for row_index, (_, row) in enumerate(chart_overall.iterrows()):
        value = float(row["Pearson Correlation"])
        y = top + row_index * row_height
        x = scale_value(min(value, 0), -1, 1, left, bar_width - right)
        end_x = scale_value(max(value, 0), -1, 1, left, bar_width - right)
        color = "#2f7d5b" if value >= 0 else "#b54d4d"
        bars.append(
            f"""
            <text x="{left - 10}" y="{y + 19}" class="svg-tick" text-anchor="end">{svg_text(row["Lagged Variable"])}</text>
            <rect x="{x:.2f}" y="{y + 5}" width="{max(end_x - x, 1):.2f}" height="20" fill="{color}" rx="3"></rect>
            <text x="{end_x + 8:.2f}" y="{y + 20}" class="svg-tick">{value:.3f}</text>
            """
        )

    line_width = 760
    line_height = 320
    line_left = 72
    line_right = 34
    line_top = 28
    line_bottom = 54
    plot_width = line_width - line_left - line_right
    plot_height = line_height - line_top - line_bottom
    years = yearly["year"].astype(float)
    current = yearly["Current_Revenue_Impact"].astype(float) / 1_000_000
    prior = yearly["Prior_Year_Revenue_Impact"].astype(float) / 1_000_000
    min_year, max_year = years.min(), years.max()
    value_low = min(current.min(), prior.min())
    value_high = max(current.max(), prior.max())
    value_low = min(value_low, 0)
    value_high = max(value_high, value_low + 1)

    def point_series(values):
        points = []
        for year, value in zip(years, values):
            x = scale_value(year, min_year, max_year, line_left, line_left + plot_width)
            y = scale_value(value, value_low, value_high, line_top + plot_height, line_top)
            points.append((x, y))
        return points

    current_points = point_series(current)
    prior_points = point_series(prior)
    current_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in current_points)
    prior_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in prior_points)
    markers = []
    for x, y in current_points:
        markers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="#587b7f"></circle>')
    for x, y in prior_points:
        markers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="#d08c60"></circle>')

    grid = []
    labels = []
    for index in range(5):
        value = value_low + (value_high - value_low) * index / 4
        y = scale_value(value, value_low, value_high, line_top + plot_height, line_top)
        grid.append(f'<line x1="{line_left}" y1="{y:.2f}" x2="{line_left + plot_width}" y2="{y:.2f}" class="svg-grid"></line>')
        labels.append(f'<text x="{line_left - 8}" y="{y + 4:.2f}" class="svg-tick" text-anchor="end">{value:.1f}M</text>')
    for year in years.astype(int):
        if year % 3 == 0 or year in [int(min_year), int(max_year)]:
            x = scale_value(year, min_year, max_year, line_left, line_left + plot_width)
            labels.append(f'<text x="{x:.2f}" y="{line_top + plot_height + 20}" class="svg-tick" text-anchor="middle">{year}</text>')

    return f"""
    <div class="calculated-grid">
        <div class="calculated-chart">
            <h3>Prior-Year Correlation With Current Revenue</h3>
            <svg class="calculated-svg" viewBox="0 0 {bar_width} {bar_height}" role="img" aria-label="Calculated lag correlation bars">
                <line x1="{zero_x:.2f}" y1="{top - 6}" x2="{zero_x:.2f}" y2="{top + row_height * len(chart_overall)}" class="svg-axis"></line>
                {''.join(bars)}
            </svg>
        </div>
        <div class="calculated-chart">
            <h3>Current vs Prior Revenue Impact, r={year_average_correlation:.3f}</h3>
            <svg class="calculated-svg" viewBox="0 0 {line_width} {line_height}" role="img" aria-label="Calculated current and prior revenue impact by year">
                <rect x="{line_left}" y="{line_top}" width="{plot_width}" height="{plot_height}" class="svg-plot-bg"></rect>
                {''.join(grid)}
                <polyline points="{current_polyline}" class="svg-line current-line"></polyline>
                <polyline points="{prior_polyline}" class="svg-line prior-line"></polyline>
                {''.join(markers)}
                <line x1="{line_left}" y1="{line_top + plot_height}" x2="{line_left + plot_width}" y2="{line_top + plot_height}" class="svg-axis"></line>
                <line x1="{line_left}" y1="{line_top}" x2="{line_left}" y2="{line_top + plot_height}" class="svg-axis"></line>
                {''.join(labels)}
                <text x="{line_left + 8}" y="{line_top + 16}" class="svg-legend-fill current-fill">Current year</text>
                <text x="{line_left + 130}" y="{line_top + 16}" class="svg-legend-fill prior-fill">Prior year</text>
            </svg>
        </div>
    </div>
    """


def build_lag_decay_data(model_df, max_lag=5):
    sorted_df = model_df[["company_id", "year", TARGET] + LAG_DECAY_FEATURES].sort_values(
        ["company_id", "year"]
    ).copy()
    rows = []

    for lag in range(max_lag + 1):
        for feature in LAG_DECAY_FEATURES:
            if lag == 0:
                paired = sorted_df[[TARGET, feature]].dropna()
                source_column = feature
            else:
                source_column = f"lag_{lag}_{feature}"
                year_column = f"lag_{lag}_year"
                paired = sorted_df[["company_id", "year", TARGET]].copy()
                paired[source_column] = sorted_df.groupby("company_id")[feature].shift(lag)
                paired[year_column] = sorted_df.groupby("company_id")["year"].shift(lag)
                paired = paired[paired[year_column] == paired["year"] - lag].dropna(
                    subset=[TARGET, source_column]
                )

            rows.append(
                {
                    "variable": feature,
                    "lag_years": lag,
                    "source_column": source_column,
                    "rows": len(paired),
                    "pearson_correlation": paired[source_column].corr(paired[TARGET]),
                }
            )

    decay = pd.DataFrame(rows)
    average = (
        decay.groupby("lag_years", as_index=False)["pearson_correlation"]
        .mean()
        .assign(variable="average", source_column="average", rows=np.nan)
    )
    return pd.concat([decay, average], ignore_index=True)


def render_lag_decay_chart(decay):
    colors = {
        "ai_investment_usd": "#2b6fa9",
        "automation_rate": "#c77c20",
        "cost_savings": "#2f7d5b",
        "deployment_count": "#b54d4d",
        "employee_ai_training_hours": "#7b559d",
        "average": "#17202a",
    }

    width = 820
    height = 390
    left = 70
    right = 190
    top = 28
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    min_lag = decay["lag_years"].min()
    max_lag = decay["lag_years"].max()
    min_value = min(0, decay["pearson_correlation"].min())
    max_value = max(0.75, decay["pearson_correlation"].max())
    series = []

    for variable in LAG_DECAY_FEATURES + ["average"]:
        chart_data = decay[decay["variable"] == variable].sort_values("lag_years")
        points = []
        circles = []
        for _, row in chart_data.iterrows():
            x = scale_value(row["lag_years"], min_lag, max_lag, left, left + plot_width)
            y = scale_value(row["pearson_correlation"], min_value, max_value, top + plot_height, top)
            points.append(f"{x:.2f},{y:.2f}")
            circles.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors[variable]}" stroke="#ffffff" stroke-width="1"></circle>'
            )
        dash = ' stroke-dasharray="6 5"' if variable == "average" else ""
        series.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[variable]}" stroke-width="2.4"{dash}></polyline>{"".join(circles)}'
        )

    grid = []
    labels = []
    for index in range(5):
        value = min_value + (max_value - min_value) * index / 4
        y = scale_value(value, min_value, max_value, top + plot_height, top)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="svg-grid"></line>')
        labels.append(f'<text x="{left - 8}" y="{y + 4:.2f}" class="svg-tick" text-anchor="end">{value:.2f}</text>')
    for lag in sorted(decay["lag_years"].dropna().unique()):
        x = scale_value(lag, min_lag, max_lag, left, left + plot_width)
        labels.append(f'<text x="{x:.2f}" y="{top + plot_height + 22}" class="svg-tick" text-anchor="middle">{int(lag)}</text>')

    legend = []
    for index, variable in enumerate(LAG_DECAY_FEATURES + ["average"]):
        y = top + index * 24
        legend.append(
            f'<rect x="{left + plot_width + 28}" y="{y - 10}" width="12" height="12" fill="{colors[variable]}"></rect>'
            f'<text x="{left + plot_width + 46}" y="{y}" class="svg-tick">{svg_text(variable)}</text>'
        )

    return f"""
    <div class="calculated-chart">
        <h3>Correlation Decay by Lag</h3>
        <svg class="calculated-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Calculated correlation decay by lag">
            <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="svg-plot-bg"></rect>
            {''.join(grid)}
            {''.join(series)}
            <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="svg-axis"></line>
            <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="svg-axis"></line>
            {''.join(labels)}
            {''.join(legend)}
            <text x="{left + plot_width / 2}" y="{height - 18}" class="svg-axis-label" text-anchor="middle">Lag in years</text>
            <text x="18" y="{top + plot_height / 2}" class="svg-axis-label" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})">Pearson correlation</text>
        </svg>
    </div>
    """


def render_lag_heatmap(decay, max_lag=4):
    heatmap_data = decay[
        (decay["variable"].isin(LAG_DECAY_FEATURES))
        & (decay["lag_years"] <= max_lag)
    ]
    pivot = heatmap_data.pivot(
        index="variable",
        columns="lag_years",
        values="pearson_correlation",
    ).loc[LAG_DECAY_FEATURES]
    cell_width = 118
    cell_height = 48
    left = 220
    top = 56
    width = left + cell_width * len(pivot.columns) + 30
    height = top + cell_height * len(pivot.index) + 34
    max_value = max(0.75, float(np.nanmax(pivot.values)))
    cells = []

    for row_index, variable in enumerate(pivot.index):
        y = top + row_index * cell_height
        cells.append(f'<text x="{left - 14}" y="{y + 30}" class="svg-tick" text-anchor="end">{svg_text(variable)}</text>')
        for column_index, lag in enumerate(pivot.columns):
            value = float(pivot.loc[variable, lag])
            x = left + column_index * cell_width
            opacity = 0.16 + 0.76 * max(value, 0) / max_value
            cells.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width - 4}" height="{cell_height - 4}" fill="#2b6fa9" opacity="{opacity:.3f}"></rect>'
                f'<text x="{x + (cell_width - 4) / 2:.2f}" y="{y + 28}" class="svg-cell-label" text-anchor="middle">{value:.3f}</text>'
            )

    headers = []
    for column_index, lag in enumerate(pivot.columns):
        x = left + column_index * cell_width + (cell_width - 4) / 2
        headers.append(f'<text x="{x:.2f}" y="34" class="svg-axis-label" text-anchor="middle">Lag {int(lag)}</text>')

    return f"""
    <div class="calculated-chart">
        <h3>Lag Strength Heatmap</h3>
        <svg class="calculated-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Calculated lag strength heatmap">
            {''.join(headers)}
            {''.join(cells)}
        </svg>
    </div>
    """
