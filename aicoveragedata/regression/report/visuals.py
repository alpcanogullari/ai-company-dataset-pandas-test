import html

import numpy as np
import pandas as pd

from .config import FEATURES
from .formatting import directional_outliers, money, number, percent, scale_value

def render_actual_vs_predicted(predictions):
    actual = predictions["Actual Revenue Impact"].astype(float)
    predicted = predictions["Predicted Revenue Impact"].astype(float)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    span = max(high - low, 1)
    low -= span * 0.03
    high += span * 0.03

    width = 820
    height = 540
    left = 78
    right = 34
    top = 34
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    scatter_points = []

    for predicted_value, actual_value in predictions[
        ["Predicted Revenue Impact", "Actual Revenue Impact"]
    ].itertuples(index=False, name=None):
        if not (np.isfinite(predicted_value) and np.isfinite(actual_value)):
            continue
        x = scale_value(predicted_value, low, high, left, left + plot_width)
        y = scale_value(actual_value, low, high, top + plot_height, top)
        scatter_points.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.8" class="scatter-point"></circle>'
        )

    grid_lines = []
    tick_labels = []
    for index in range(5):
        value = low + (high - low) * index / 4
        x = scale_value(value, low, high, left, left + plot_width)
        y = scale_value(value, low, high, top + plot_height, top)
        grid_lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" class="svg-grid"></line>')
        grid_lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="svg-grid"></line>')
        tick_labels.append(f'<text x="{x:.2f}" y="{top + plot_height + 22}" class="svg-tick" text-anchor="middle">{money(value)}</text>')
        tick_labels.append(f'<text x="{left - 10}" y="{y + 4:.2f}" class="svg-tick" text-anchor="end">{money(value)}</text>')

    diagonal = (
        f'<line x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top}" class="svg-diagonal"></line>'
    )

    top_under, top_over = directional_outliers(predictions)
    markers = []
    marker_groups = [
        ("U", top_under, "#d04a3a", "circle"),
        ("O", top_over, "#2f6fbb", "square"),
    ]
    for label_prefix, rows, color, marker_shape in marker_groups:
        for _, row in rows.sort_values("Outlier Rank").iterrows():
            x = scale_value(row["Predicted Revenue Impact"], low, high, left, left + plot_width)
            y = scale_value(row["Actual Revenue Impact"], low, high, top + plot_height, top)
            label = f"{label_prefix}{int(row['Outlier Rank'])}"
            if marker_shape == "square":
                shape = f'<rect x="{x - 6:.2f}" y="{y - 6:.2f}" width="12" height="12" fill="{color}" stroke="#ffffff" stroke-width="1.5"></rect>'
            else:
                shape = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"></circle>'
            markers.append(
                f'{shape}<text x="{x + 8:.2f}" y="{y - 8:.2f}" class="svg-marker-label" fill="{color}">{label}</text>'
            )

    return f"""
    <div class="calculated-chart">
        <h3>Actual vs Predicted Revenue Impact</h3>
            <svg class="calculated-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Calculated actual versus predicted revenue impact">
            <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="svg-plot-bg"></rect>
            {''.join(grid_lines)}
            <g class="scatter-points">{''.join(scatter_points)}</g>
            {diagonal}
            {''.join(markers)}
            <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="svg-axis"></line>
            <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="svg-axis"></line>
            {''.join(tick_labels)}
            <text x="{left + plot_width / 2}" y="{height - 18}" class="svg-axis-label" text-anchor="middle">Predicted Revenue Impact</text>
            <text x="18" y="{top + plot_height / 2}" class="svg-axis-label" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})">Actual Revenue Impact</text>
        </svg>
        <div class="chart-caption">Calculated from all {len(predictions):,} held-out test rows as a scatter plot. The dashed line is actual = predicted.</div>
    </div>
    """


def outlier_cards(rows, label_prefix):
    tone = "under" if label_prefix == "U" else "over"
    cards = []
    for _, row in rows.sort_values("Outlier Rank").iterrows():
        vars_html = "\n".join(
            f"<span>{feature}: {number(row[feature])}</span>" for feature in FEATURES
        )
        cards.append(
            f"""
            <div class="outlier-item {tone}">
                <strong>{label_prefix}{int(row["Outlier Rank"])} {html.escape(row["Error Direction"])}</strong>
                <span>Actual {money(row["Actual Revenue Impact"])}</span>
                <span>Predicted {money(row["Predicted Revenue Impact"])}</span>
                <span>Error {money(row["Error"])} ({percent(row["Error Percentage"])})</span>
                <div class="outlier-vars">
                    <b>Independent variables</b>
                    {vars_html}
                </div>
            </div>
            """
        )
    return "".join(cards)


def svg_marker_x(value, low, high, width):
    if high <= low:
        return width / 2
    position = (value - low) / (high - low) * width
    return min(max(position, 0), width)


def skewness_distribution_cards(test_data, skewness, title, columns):
    cards = []
    lookup = skewness.set_index("Variable")
    svg_width = 360
    svg_height = 112
    plot_height = 88
    bins = 34

    for column in columns:
        values = pd.to_numeric(test_data[column], errors="coerce").dropna()
        if values.empty or column not in lookup.index:
            continue

        low = values.quantile(0.01)
        high = values.quantile(0.99)
        if high <= low:
            low = values.min()
            high = values.max()
        if high <= low:
            high = low + 1

        clipped = values.clip(lower=low, upper=high)
        counts, edges = np.histogram(clipped, bins=bins, range=(low, high))
        max_count = max(int(counts.max()), 1)
        bar_width = svg_width / bins
        rects = []
        for index, count in enumerate(counts):
            height = count / max_count * plot_height
            x = index * bar_width
            y = plot_height - height
            rects.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 1, 1):.2f}" height="{height:.2f}" rx="1"></rect>'
            )

        mean = values.mean()
        median = values.median()
        mean_x = svg_marker_x(mean, low, high, svg_width)
        median_x = svg_marker_x(median, low, high, svg_width)
        skewness_value = float(lookup.loc[column, "Skewness"])
        direction = lookup.loc[column, "Skew Direction"]
        strength = lookup.loc[column, "Skew Strength"]
        clipped_note = "central 98% shown"
        percentile_rows = [
            ("P1", values.quantile(0.01)),
            ("P10", values.quantile(0.10)),
            ("P25", values.quantile(0.25)),
            ("Mean", mean),
            ("P50", values.quantile(0.50)),
            ("P75", values.quantile(0.75)),
            ("P90", values.quantile(0.90)),
            ("P99", values.quantile(0.99)),
        ]
        percentile_html = "".join(
            f"<span><b>{label}</b>{number(value)}</span>"
            for label, value in percentile_rows
        )

        cards.append(
            f"""
            <div class="skew-card">
                <div class="skew-card-head">
                    <strong>{html.escape(column)}</strong>
                    <span>{html.escape(direction)} / {html.escape(strength)} / skew {skewness_value:.3f}</span>
                </div>
                <svg class="skew-svg" viewBox="0 0 {svg_width} {svg_height}" role="img" aria-label="{html.escape(column)} distribution">
                    <g class="skew-bars">{''.join(rects)}</g>
                    <line class="median-marker" x1="{median_x:.2f}" y1="0" x2="{median_x:.2f}" y2="{plot_height}"></line>
                    <line class="mean-marker" x1="{mean_x:.2f}" y1="0" x2="{mean_x:.2f}" y2="{plot_height}"></line>
                    <line class="axis-line" x1="0" y1="{plot_height}" x2="{svg_width}" y2="{plot_height}"></line>
                </svg>
                <div class="skew-axis">
                    <span>{number(low)}</span>
                    <span>{html.escape(clipped_note)}</span>
                    <span>{number(high)}</span>
                </div>
                <div class="skew-legend">
                    <span><i class="median-key"></i>Median {number(median)}</span>
                    <span><i class="mean-key"></i>Mean {number(mean)}</span>
                </div>
                <div class="percentile-strip">{percentile_html}</div>
            </div>
            """
        )

    return f"""
    <div class="skew-distribution">
        <h3>{html.escape(title)}</h3>
        <div class="skew-card-grid">{''.join(cards)}</div>
    </div>
    """
