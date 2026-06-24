import html

import numpy as np
import pandas as pd


def money(value):
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1_000_000:
        return f"{sign}{value / 1_000_000:,.2f}M"
    if value >= 1_000:
        return f"{sign}{value / 1_000:,.2f}K"
    return f"{sign}{value:,.2f}"


def number(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1_000:
            return f"{value:,.2f}"
        return f"{value:,.4f}"
    return html.escape(str(value))


def percent(value):
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value):,.2f}%"


def adjusted_r2(r2, rows, feature_count):
    return 1 - ((1 - r2) * (rows - 1) / (rows - feature_count - 1))


def table_html(frame):
    return frame.to_html(index=False, classes="excel-table", border=0, escape=False)


def error_direction(error):
    if error > 0:
        return "Underpredicted"
    if error < 0:
        return "Overpredicted"
    return "Exact"


def add_error_percentages(frame):
    result = frame.copy()
    actual = result["Actual Revenue Impact"].astype(float)
    denominator = actual.abs()
    result["Error Percentage"] = np.where(
        denominator != 0,
        result["Error"].astype(float) / denominator * 100,
        np.nan,
    )
    result["Absolute Error Percentage"] = result["Error Percentage"].abs()
    return result


def directional_outliers(predictions, limit=5):
    enriched = add_error_percentages(predictions)
    top_under = enriched[enriched["Error"] > 0].nlargest(limit, "Absolute Error").copy()
    top_over = enriched[enriched["Error"] < 0].nlargest(limit, "Absolute Error").copy()
    top_under["Outlier Rank"] = range(1, len(top_under) + 1)
    top_over["Outlier Rank"] = range(1, len(top_over) + 1)
    return top_under, top_over


def scale_value(value, low, high, start, end):
    if high <= low or pd.isna(value):
        return (start + end) / 2
    ratio = (float(value) - float(low)) / (float(high) - float(low))
    return start + min(max(ratio, 0), 1) * (end - start)


def svg_text(value):
    return html.escape(str(value))
