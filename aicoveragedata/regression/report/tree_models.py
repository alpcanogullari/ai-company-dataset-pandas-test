import html

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .config import (
    ADOPTION_FEATURES,
    ADOPTION_TARGET,
    FEATURES,
    TARGET,
    TREE_MAX_DEPTH,
    TREE_MIN_NODE_GAIN,
    TREE_MIN_SAMPLES_LEAF,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_TREES,
)
from .formatting import error_direction, money, number, scale_value, svg_text
from .modeling import build_metrics, metrics_table
from .profiles import build_feature_profile


def build_feature_profile_for_columns(frame, feature_columns):
    rows = []
    for feature in feature_columns:
        series = pd.to_numeric(frame[feature], errors="coerce").dropna()
        skewness = series.skew() if len(series) > 2 else np.nan
        if pd.isna(skewness):
            skew_direction = "Not available"
            skew_strength = "Not available"
        else:
            skew_direction = "Right-skewed" if skewness > 0 else "Left-skewed" if skewness < 0 else "Symmetric"
            magnitude = abs(skewness)
            skew_strength = "Low" if magnitude < 0.5 else "Moderate" if magnitude < 1 else "High"
        rows.append(
            {
                "Variable": feature,
                "Rows": len(series),
                "P1": series.quantile(0.01),
                "P10": series.quantile(0.10),
                "P25": series.quantile(0.25),
                "Mean": series.mean(),
                "P50": series.quantile(0.50),
                "P75": series.quantile(0.75),
                "P90": series.quantile(0.90),
                "P99": series.quantile(0.99),
                "Std Dev": series.std(),
                "Min": series.min(),
                "Max": series.max(),
                "Skewness": skewness,
                "Skew Direction": skew_direction,
                "Skew Strength": skew_strength,
            }
        )
    return pd.DataFrame(rows)


def build_xgboost_inputs_for_features(base_df, feature_columns, feature_profile):
    tree_df = base_df.copy()
    profile = feature_profile.set_index("Variable")
    tree_features = []
    transform_lookup = {}
    feature_rows = []

    for feature in feature_columns:
        skewness = float(profile.loc[feature, "Skewness"])
        skew_direction_value = profile.loc[feature, "Skew Direction"]
        skew_strength_value = profile.loc[feature, "Skew Strength"]

        if skewness > 1 and tree_df[feature].min() >= 0:
            tree_feature = f"log1p_{feature}"
            tree_df[tree_feature] = np.log1p(tree_df[feature])
            transform = "log1p (high right skew)"
        else:
            tree_feature = feature
            transform = "raw"

        tree_features.append(tree_feature)
        transform_lookup[tree_feature] = {
            "original": feature,
            "transform": transform,
            "skewness": skewness,
            "skew_direction": skew_direction_value,
            "skew_strength": skew_strength_value,
        }
        feature_rows.append(
            {
                "Original Feature": feature,
                "Tree Feature": tree_feature,
                "Transform": transform,
                "Skewness": skewness,
                "Skew Direction": skew_direction_value,
                "Skew Strength": skew_strength_value,
                "Min": profile.loc[feature, "Min"],
                "P1": profile.loc[feature, "P1"],
                "P10": profile.loc[feature, "P10"],
                "P25": profile.loc[feature, "P25"],
                "Mean": profile.loc[feature, "Mean"],
                "P50": profile.loc[feature, "P50"],
                "P75": profile.loc[feature, "P75"],
                "P90": profile.loc[feature, "P90"],
                "P99": profile.loc[feature, "P99"],
                "Max": profile.loc[feature, "Max"],
            }
        )

    return tree_df, tree_features, transform_lookup, pd.DataFrame(feature_rows)

def tree_original_threshold(tree_feature, threshold, transform_lookup):
    transform = transform_lookup[tree_feature]["transform"]
    if transform.startswith("log1p"):
        return np.expm1(threshold)
    return threshold


def build_decision_tree_inputs(model_df, feature_percentiles):
    tree_df = model_df[["company_id", "year"] + FEATURES + [TARGET]].dropna().copy()
    profile = feature_percentiles.set_index("Variable")
    tree_features = []
    transform_lookup = {}
    feature_rows = []

    for feature in FEATURES:
        skewness = float(profile.loc[feature, "Skewness"])
        skew_direction_value = profile.loc[feature, "Skew Direction"]
        skew_strength_value = profile.loc[feature, "Skew Strength"]

        if skewness > 1 and tree_df[feature].min() >= 0:
            tree_feature = f"log1p_{feature}"
            tree_df[tree_feature] = np.log1p(tree_df[feature])
            transform = "log1p (high right skew)"
        else:
            tree_feature = feature
            transform = "raw"

        tree_features.append(tree_feature)
        transform_lookup[tree_feature] = {
            "original": feature,
            "transform": transform,
            "skewness": skewness,
            "skew_direction": skew_direction_value,
            "skew_strength": skew_strength_value,
        }
        feature_rows.append(
            {
                "Original Feature": feature,
                "Tree Feature": tree_feature,
                "Transform": transform,
                "Skewness": skewness,
                "Skew Direction": skew_direction_value,
                "Skew Strength": skew_strength_value,
                "Min": profile.loc[feature, "Min"],
                "P1": profile.loc[feature, "P1"],
                "P10": profile.loc[feature, "P10"],
                "P25": profile.loc[feature, "P25"],
                "Mean": profile.loc[feature, "Mean"],
                "P50": profile.loc[feature, "P50"],
                "P75": profile.loc[feature, "P75"],
                "P90": profile.loc[feature, "P90"],
                "P99": profile.loc[feature, "P99"],
                "Max": profile.loc[feature, "Max"],
            }
        )

    return tree_df, tree_features, transform_lookup, pd.DataFrame(feature_rows)


def sklearn_node_variance_gain(model, node_id):
    tree = model.tree_
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child == right_child:
        return ""

    parent_sse = tree.impurity[node_id] * tree.n_node_samples[node_id]
    if parent_sse <= 0:
        return 0
    child_sse = (
        tree.impurity[left_child] * tree.n_node_samples[left_child]
        + tree.impurity[right_child] * tree.n_node_samples[right_child]
    )
    return max(float((parent_sse - child_sse) / parent_sse), 0)


def sklearn_node_sse_gain(model, node_id):
    tree = model.tree_
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child == right_child:
        return 0

    parent_sse = tree.impurity[node_id] * tree.n_node_samples[node_id]
    child_sse = (
        tree.impurity[left_child] * tree.n_node_samples[left_child]
        + tree.impurity[right_child] * tree.n_node_samples[right_child]
    )
    return max(float(parent_sse - child_sse), 0)


def pruned_tree_is_leaf(model, node_id):
    tree = model.tree_
    if tree.children_left[node_id] == tree.children_right[node_id]:
        return True
    return sklearn_node_variance_gain(model, node_id) <= TREE_MIN_NODE_GAIN


def pruned_tree_stop_reason(model, node_id):
    tree = model.tree_
    if tree.children_left[node_id] == tree.children_right[node_id]:
        return "Original leaf"
    gain = sklearn_node_variance_gain(model, node_id)
    if gain <= TREE_MIN_NODE_GAIN:
        return f"Node variance gain {gain:.2%} <= 5%"
    return ""


def predict_pruned_decision_tree(model, X):
    tree = model.tree_
    values = X.to_numpy(dtype=float)
    predictions = np.empty(len(values), dtype=float)

    for row_index, row in enumerate(values):
        node_id = 0
        while not pruned_tree_is_leaf(model, node_id):
            feature_index = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            node_id = (
                tree.children_left[node_id]
                if row[feature_index] <= threshold
                else tree.children_right[node_id]
            )
        predictions[row_index] = tree.value[node_id][0][0]

    return predictions


def pruned_tree_depth(model, node_id=0, depth=0):
    if pruned_tree_is_leaf(model, node_id):
        return depth
    tree = model.tree_
    return max(
        pruned_tree_depth(model, tree.children_left[node_id], depth + 1),
        pruned_tree_depth(model, tree.children_right[node_id], depth + 1),
    )


def pruned_tree_leaf_count(model, node_id=0):
    if pruned_tree_is_leaf(model, node_id):
        return 1
    tree = model.tree_
    return (
        pruned_tree_leaf_count(model, tree.children_left[node_id])
        + pruned_tree_leaf_count(model, tree.children_right[node_id])
    )


def pruned_tree_importance(model, tree_features, transform_lookup):
    tree = model.tree_
    scores = {feature: 0.0 for feature in tree_features}

    def visit(node_id):
        if pruned_tree_is_leaf(model, node_id):
            return
        tree_feature = tree_features[tree.feature[node_id]]
        scores[tree_feature] += sklearn_node_sse_gain(model, node_id)
        visit(tree.children_left[node_id])
        visit(tree.children_right[node_id])

    visit(0)
    total = sum(scores.values())
    rows = []
    for tree_feature in tree_features:
        info = transform_lookup[tree_feature]
        rows.append(
            {
                "Original Feature": info["original"],
                "Tree Feature": tree_feature,
                "Transform": info["transform"],
                "Importance": scores[tree_feature] / total if total else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("Importance", ascending=False)


def build_sklearn_decision_tree_rules(model, tree_features, transform_lookup):
    tree = model.tree_
    rows = []

    def visit(node_id, depth, path):
        is_leaf = pruned_tree_is_leaf(model, node_id)
        tree_feature = ""
        original_feature = ""
        original_threshold = ""
        variance_gain = ""
        gain_threshold = ""
        stop_reason = pruned_tree_stop_reason(model, node_id)
        skewness = ""
        skew_direction_value = ""

        if tree.children_left[node_id] != tree.children_right[node_id]:
            tree_feature = tree_features[tree.feature[node_id]]
            original_threshold = tree_original_threshold(
                tree_feature,
                float(tree.threshold[node_id]),
                transform_lookup,
            )
            info = transform_lookup[tree_feature]
            original_feature = info["original"]
            skewness = info["skewness"]
            skew_direction_value = info["skew_direction"]
            variance_gain = sklearn_node_variance_gain(model, node_id)
            gain_threshold = TREE_MIN_NODE_GAIN

        rows.append(
            {
                "Node": node_id,
                "Depth": depth,
                "Node Type": "Leaf" if is_leaf else "Split",
                "Rule Path": " and ".join(path) if path else "Root",
                "Split Feature": original_feature,
                "Tree Feature": tree_feature,
                "Original Threshold": original_threshold,
                "Rows": int(tree.n_node_samples[node_id]),
                "Node Prediction": float(tree.value[node_id][0][0]),
                "Node Variance Gain": variance_gain,
                "Gain Threshold": gain_threshold,
                "Stop Reason": stop_reason,
                "Skewness": skewness,
                "Skew Direction": skew_direction_value,
            }
        )

        if is_leaf:
            return

        left_condition = f"{original_feature} <= {number(original_threshold)}"
        right_condition = f"{original_feature} > {number(original_threshold)}"
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        visit(left_child, depth + 1, path + [left_condition])
        visit(right_child, depth + 1, path + [right_condition])

    visit(0, 0, [])
    return pd.DataFrame(rows)


def wrap_tree_label(value, max_chars=22):
    parts = str(value).split("_")
    lines = []
    current = ""
    for part in parts:
        candidate = f"{current}_{part}" if current else part
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:2]


def clean_float(value):
    if value == "" or pd.isna(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def svg_line_block(x, y, lines, css_class, line_height=13):
    text = []
    for line in lines:
        text.append(
            f'<text x="{x:.2f}" y="{y:.2f}" class="{css_class}" text-anchor="middle">{svg_text(line)}</text>'
        )
        y += line_height
    return "".join(text)


def render_decision_tree(
    model,
    tree_features,
    transform_lookup,
    title="Single Pruned Decision Tree",
    subtitle="The single tree is the readable threshold view. It stops a branch when the next split adds 5% or less node variance gain.",
    prediction_formatter=money,
):
    tree = model.tree_
    node_width = 168
    node_height = 120
    x_gap = 38
    y_gap = 34
    left_margin = 30
    top = 36
    positions = {}
    leaf_index = 0

    def layout(node_id, depth):
        nonlocal leaf_index
        if pruned_tree_is_leaf(model, node_id):
            x = left_margin + node_width / 2 + leaf_index * (node_width + x_gap)
            leaf_index += 1
        else:
            left_x = layout(tree.children_left[node_id], depth + 1)
            right_x = layout(tree.children_right[node_id], depth + 1)
            x = (left_x + right_x) / 2
        positions[node_id] = (x, depth)
        return x

    layout(0, 0)
    max_depth = max(depth for _, depth in positions.values())
    width = max(
        900,
        left_margin * 2 + leaf_index * node_width + max(leaf_index - 1, 0) * x_gap,
    )
    height = top + max_depth * (node_height + y_gap) + node_height + 36
    links = []
    nodes = []

    for node_id, (x, depth) in sorted(positions.items(), key=lambda item: (item[1][1], item[1][0])):
        y = top + depth * (node_height + y_gap)
        if not pruned_tree_is_leaf(model, node_id):
            for child_id in [tree.children_left[node_id], tree.children_right[node_id]]:
                child_x, child_depth = positions[child_id]
                child_y = top + child_depth * (node_height + y_gap)
                links.append(
                    f'<line x1="{x:.2f}" y1="{y + node_height:.2f}" x2="{child_x:.2f}" y2="{child_y:.2f}" class="tree-link"></line>'
                )

    for node_id, (x, depth) in sorted(positions.items(), key=lambda item: (item[1][1], item[1][0])):
        y = top + depth * (node_height + y_gap)
        is_leaf = pruned_tree_is_leaf(model, node_id)
        node_class = "leaf" if is_leaf else "split"
        text_y = y + 21

        if is_leaf:
            title_lines = ["Leaf"]
            detail_lines = []
            if tree.children_left[node_id] != tree.children_right[node_id]:
                detail_lines.append("gain <= 5%")
            detail_lines.extend(
                [
                    f"n={int(tree.n_node_samples[node_id]):,}",
                    f"pred {prediction_formatter(tree.value[node_id][0][0])}",
                ]
            )
        else:
            tree_feature = tree_features[tree.feature[node_id]]
            info = transform_lookup[tree_feature]
            original_threshold = tree_original_threshold(
                tree_feature,
                float(tree.threshold[node_id]),
                transform_lookup,
            )
            title_lines = wrap_tree_label(info["original"])
            detail_lines = [
                f"<= {number(original_threshold)}",
                f"gain {sklearn_node_variance_gain(model, node_id):.1%}",
                f"n={int(tree.n_node_samples[node_id]):,}",
                f"pred {prediction_formatter(tree.value[node_id][0][0])}",
                f"skew {float(info['skewness']):.2f}, {info['skew_strength']}",
            ]

        text = []
        for line in title_lines:
            text.append(
                f'<text x="{x:.2f}" y="{text_y:.2f}" class="tree-node-title" text-anchor="middle">{svg_text(line)}</text>'
            )
            text_y += 14
        for line in detail_lines:
            text.append(
                f'<text x="{x:.2f}" y="{text_y:.2f}" class="tree-node-text" text-anchor="middle">{svg_text(line)}</text>'
            )
            text_y += 14

        nodes.append(
            f"""
            <g>
                <rect x="{x - node_width / 2:.2f}" y="{y:.2f}" width="{node_width}" height="{node_height}" rx="7" class="tree-node {node_class}"></rect>
                {''.join(text)}
            </g>
            """
        )

    return f"""
    <div class="calculated-chart tree-scroll">
        <h3>{html.escape(title)}</h3>
        <p>{html.escape(subtitle)}</p>
        <svg class="calculated-svg tree-svg" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Calculated skew-aware decision tree">
            {''.join(links)}
            {''.join(nodes)}
        </svg>
    </div>
    """


def render_xgboost_ensemble_visual(importance, rules, split_audit, model_selection):
    r2_rows = model_selection[
        model_selection["Check"].str.contains("Test R Square", na=False)
    ].copy()
    max_r2 = max(pd.to_numeric(r2_rows["Value"], errors="coerce").max(), 1)
    score_rows = []
    for _, row in r2_rows.iterrows():
        label = row["Check"].replace(" Test R Square", "")
        value = float(row["Value"])
        width = max(value / max_r2 * 100, 1)
        score_rows.append(
            f"""
            <div class="model-score-row">
                <strong>{html.escape(label)}</strong>
                <span class="model-score-track"><i style="width:{width:.1f}%"></i></span>
                <b>{value:.4f}</b>
            </div>
            """
        )

    split_rows = rules[rules["Node Type"] == "Split"].copy()
    split_counts = split_rows.groupby("Original Feature").size().to_dict()
    total_splits = int(len(split_rows))
    audit_lookup = split_audit.set_index("Check")["Value"].to_dict()
    minimum_gain = audit_lookup.get("Minimum Native Gain Threshold", "")

    feature_colors = {
        "cost_savings": "#2f6fbb",
        "ai_investment_usd": "#587b7f",
        "employee_ai_training_hours": "#d08c60",
        "deployment_count": "#7a5c9e",
        "automation_rate": "#6c8f3d",
    }
    fallback_colors = ["#2f6fbb", "#587b7f", "#d08c60", "#7a5c9e", "#6c8f3d"]
    for index, feature in enumerate(importance["Original Feature"]):
        feature_colors.setdefault(feature, fallback_colors[index % len(fallback_colors)])

    bar_rows = []
    for _, row in importance.iterrows():
        feature = row["Original Feature"]
        share = float(row["Importance"])
        count = int(split_counts.get(feature, 0))
        color = feature_colors[feature]
        bar_rows.append(
            f"""
            <div class="xgb-importance-row">
                <strong>{html.escape(feature)}</strong>
                <span class="xgb-importance-track"><i style="width:{share * 100:.1f}%; background:{color}"></i></span>
                <b>{share * 100:.1f}%</b>
                <em>{count:,} splits</em>
            </div>
            """
        )

    tree_feature = {}
    for tree_id, tree_rows in split_rows.groupby("Tree"):
        gain_by_feature = tree_rows.groupby("Original Feature")["Native Gain"].sum()
        if not gain_by_feature.empty:
            tree_feature[int(tree_id)] = str(gain_by_feature.idxmax())

    tree_tiles = []
    tree_count = int(max(rules["Tree"].max(), -1) + 1) if not rules.empty else 0
    for tree_id in range(tree_count):
        feature = tree_feature.get(tree_id, "leaf-only")
        color = feature_colors.get(feature, "#c6cbd1")
        label = feature.replace("_", " ")
        tree_tiles.append(
            f'<span class="xgb-tree-tile" style="background:{color}" title="Tree {tree_id}: {html.escape(label)}"></span>'
        )

    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(feature)}</span>'
        for feature, color in feature_colors.items()
        if feature in set(importance["Original Feature"])
    )

    return f"""
    <div class="calculated-chart">
        <h3>XGBoost Ensemble Visual</h3>
        <p>XGBoost is not one tree. It is an ensemble of {tree_count:,} shallow trees. Each tree adds a small correction, and the combined model slightly improves held-out R Square.</p>
        <div class="model-score-visual">{''.join(score_rows)}</div>
        <div class="xgb-visual-grid">
            <div class="xgb-visual-panel">
                <h4>Feature importance and split use</h4>
                <div class="xgb-importance-list">{''.join(bar_rows)}</div>
            </div>
            <div class="xgb-visual-panel">
                <h4>Forest map: dominant feature by tree</h4>
                <div class="xgb-tree-grid">{''.join(tree_tiles)}</div>
                <div class="xgb-feature-legend">{legend}</div>
            </div>
        </div>
        <div class="split-metrics xgb-audit-strip">
            <span><b>Ensemble split rows</b>{total_splits:,}</span>
            <span><b>Minimum native gain</b>{number(minimum_gain)}</span>
            <span><b>Stop rule</b>gamma = 5% root SSE / tree count</span>
            <span><b>Reading</b>cost savings dominates; AI investment is the secondary signal</span>
        </div>
    </div>
    """


def render_feature_range_overview(feature_map, rules):
    used_features = set(
        rules.loc[rules["Node Type"] == "Split", "Split Feature"]
        .replace("", np.nan)
        .dropna()
    )
    cards = []

    for _, row in feature_map.iterrows():
        feature = row["Original Feature"]
        status = "Used in interaction tree" if feature in used_features else "Shown in separate checks"
        cards.append(
            f"""
            <div class="split-card">
                <div class="split-card-head">
                    <strong>{html.escape(feature)}</strong>
                    <span>{html.escape(status)}</span>
                </div>
                <div class="split-metrics">
                    <span><b>Full range</b>{number(row["Min"])} to {number(row["Max"])}</span>
                    <span><b>Central 98%</b>{number(row["P1"])} to {number(row["P99"])}</span>
                </div>
                <div class="split-metrics">
                    <span><b>Median</b>{number(row["P50"])}</span>
                    <span><b>Transform</b>{html.escape(row["Transform"])}</span>
                </div>
            </div>
            """
        )

    return f"""
    <div class="calculated-chart">
        <h3>Full Variable Range Overview</h3>
        <p>The interaction tree only displays variables it selected. This section keeps every model variable visible.</p>
        <div class="split-card-grid">{''.join(cards)}</div>
    </div>
    """


def render_separated_feature_tree_visual(separated_splits, feature_map):
    range_lookup = feature_map.set_index("Original Feature").to_dict("index")
    cards = []

    for _, row in separated_splits.iterrows():
        feature = row["Variable"]
        range_info = range_lookup.get(feature, {})
        min_value = clean_float(range_info.get("Min", ""))
        p1_value = clean_float(range_info.get("P1", ""))
        median_value = clean_float(range_info.get("P50", ""))
        p99_value = clean_float(range_info.get("P99", ""))
        max_value = clean_float(range_info.get("Max", ""))
        threshold = clean_float(row["Threshold"])
        variance_gain = clean_float(row["Node Variance Gain"])
        test_r2 = clean_float(row["Test R Square"])
        split_exists = row["Status"] == "Split" and threshold is not None

        width = 520
        height = 282
        root_x = 260
        root_y = 18
        node_width = 170
        node_height = 82
        leaf_y = 134
        left_x = 140
        right_x = 380
        bar_x = 46
        bar_y = 238
        bar_width = 428

        def bar_position(value):
            if min_value is None or max_value is None or max_value <= min_value:
                return bar_x + bar_width / 2
            return scale_value(value, min_value, max_value, bar_x, bar_x + bar_width)

        central_range = ""
        if p1_value is not None and p99_value is not None:
            central_x = bar_position(p1_value)
            central_width = max(bar_position(p99_value) - central_x, 2)
            central_range = (
                f'<rect x="{central_x:.2f}" y="{bar_y:.2f}" width="{central_width:.2f}" '
                'height="10" fill="#7aa0a6" opacity="0.65"></rect>'
            )

        median_marker = ""
        if median_value is not None:
            median_x = bar_position(median_value)
            median_marker = (
                f'<line x1="{median_x:.2f}" y1="{bar_y - 5:.2f}" x2="{median_x:.2f}" '
                f'y2="{bar_y + 17:.2f}" class="median-marker"></line>'
            )

        threshold_marker = ""
        if split_exists:
            threshold_x = bar_position(threshold)
            threshold_marker = f"""
                <line x1="{threshold_x:.2f}" y1="{bar_y - 14:.2f}" x2="{threshold_x:.2f}" y2="{bar_y + 22:.2f}" class="prior-line"></line>
            """

        min_label = number(min_value) if min_value is not None else ""
        max_label = number(max_value) if max_value is not None else ""
        gain_label = "" if variance_gain is None else f"gain {variance_gain:.1%}"
        r2_label = "" if test_r2 is None else f"test R2 {test_r2:.3f}"

        links = ""
        node_markup = ""
        if split_exists:
            links = f"""
                <line x1="{root_x:.2f}" y1="{root_y + node_height:.2f}" x2="{left_x:.2f}" y2="{leaf_y:.2f}" class="tree-link"></line>
                <line x1="{root_x:.2f}" y1="{root_y + node_height:.2f}" x2="{right_x:.2f}" y2="{leaf_y:.2f}" class="tree-link"></line>
            """
            root_text = svg_line_block(
                root_x,
                root_y + 22,
                wrap_tree_label(feature, 20),
                "tree-node-title",
            )
            root_text += svg_line_block(
                root_x,
                root_y + 22 + 13 * len(wrap_tree_label(feature, 20)),
                [f"<= {number(threshold)}", gain_label, r2_label],
                "tree-node-text",
            )
            left_text = svg_line_block(
                left_x,
                leaf_y + 25,
                [
                    "Leaf: <=",
                    f"n={int(row['Left Rows']):,}",
                    f"pred {money(row['Left Prediction'])}",
                ],
                "tree-node-text",
            )
            right_text = svg_line_block(
                right_x,
                leaf_y + 25,
                [
                    "Leaf: >",
                    f"n={int(row['Right Rows']):,}",
                    f"pred {money(row['Right Prediction'])}",
                ],
                "tree-node-text",
            )
            node_markup = f"""
                <rect x="{root_x - node_width / 2:.2f}" y="{root_y:.2f}" width="{node_width}" height="{node_height}" rx="7" class="tree-node split"></rect>
                {root_text}
                <rect x="{left_x - node_width / 2:.2f}" y="{leaf_y:.2f}" width="{node_width}" height="{node_height}" rx="7" class="tree-node leaf"></rect>
                {left_text}
                <rect x="{right_x - node_width / 2:.2f}" y="{leaf_y:.2f}" width="{node_width}" height="{node_height}" rx="7" class="tree-node leaf"></rect>
                {right_text}
            """
        else:
            stop_rule = str(row["Stop Rule"]) if row["Stop Rule"] else "No accepted split"
            root_text = svg_line_block(
                root_x,
                root_y + 26,
                wrap_tree_label(feature, 20),
                "tree-node-title",
            )
            root_text += svg_line_block(
                root_x,
                root_y + 26 + 13 * len(wrap_tree_label(feature, 20)),
                ["Leaf", stop_rule, r2_label],
                "tree-node-text",
            )
            node_markup = f"""
                <rect x="{root_x - node_width / 2:.2f}" y="{root_y:.2f}" width="{node_width}" height="{node_height + 8}" rx="7" class="tree-node leaf"></rect>
                {root_text}
            """

        cards.append(
            f"""
            <div class="split-card">
                <div class="split-card-head">
                    <strong>{html.escape(feature)}</strong>
                    <span>one-variable tree / {html.escape(row["Transform"])}</span>
                </div>
                <svg class="calculated-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(feature)} one-variable decision tree">
                    {links}
                    {node_markup}
                    <text x="{bar_x:.2f}" y="{bar_y - 18:.2f}" class="tree-node-text">full range</text>
                    <rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="10" fill="#eef3f7" stroke="#d8dee4"></rect>
                    {central_range}
                    {median_marker}
                    {threshold_marker}
                    <text x="{bar_x:.2f}" y="{bar_y + 34:.2f}" class="tree-node-text" text-anchor="start">{svg_text(min_label)}</text>
                    <text x="{bar_x + bar_width:.2f}" y="{bar_y + 34:.2f}" class="tree-node-text" text-anchor="end">{svg_text(max_label)}</text>
                </svg>
            </div>
            """
        )

    return f"""
    <div class="calculated-chart">
        <h3>One-Variable Tree Visuals</h3>
        <p>Each mini tree tests one variable alone, so the full feature range is visible even when the interaction tree skips that variable.</p>
        <div class="split-card-grid">{''.join(cards)}</div>
    </div>
    """


def fit_decision_tree(model_df):
    base_df = model_df[["company_id", "year"] + FEATURES + [TARGET]].dropna().copy()
    raw_X = base_df[FEATURES]
    y = base_df[TARGET]
    raw_X_train, raw_X_test, y_train, y_test = train_test_split(
        raw_X,
        y,
        test_size=0.30,
        random_state=42,
    )
    tree_profile = build_feature_profile(raw_X_train)
    tree_df, tree_features, transform_lookup, feature_map = build_decision_tree_inputs(
        base_df,
        tree_profile,
    )
    X = tree_df[tree_features]
    X_train = X.loc[raw_X_train.index]
    X_test = X.loc[raw_X_test.index]

    model = DecisionTreeRegressor(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = predict_pruned_decision_tree(model, X_train)
    test_pred = predict_pruned_decision_tree(model, X_test)
    metrics = build_metrics(y_train, train_pred, y_test, test_pred, len(tree_features))
    stats = metrics_table(metrics)
    stats.loc[len(stats)] = {
        "Metric": "Max Depth",
        "Training Data": pruned_tree_depth(model),
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Leaves",
        "Training Data": pruned_tree_leaf_count(model),
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Minimum Leaf Rows",
        "Training Data": TREE_MIN_SAMPLES_LEAF,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Leaf Stop Rule",
        "Training Data": "end leaf if node variance gain <= 5%",
        "Test Data": "",
    }

    importance = pruned_tree_importance(model, tree_features, transform_lookup)

    train_predictions = tree_df.loc[X_train.index, ["company_id", "year"] + FEATURES].copy()
    train_predictions["Case ID"] = X_train.index
    train_predictions["Actual Revenue Impact"] = y_train
    train_predictions["Decision Tree Prediction"] = train_pred
    train_predictions["Error"] = y_train - train_pred
    train_predictions["Absolute Error"] = train_predictions["Error"].abs()
    train_predictions["Error Direction"] = train_predictions["Error"].map(error_direction)
    train_predictions = train_predictions.sort_values("Case ID")

    predictions = tree_df.loc[X_test.index, ["company_id", "year"] + FEATURES].copy()
    predictions["Case ID"] = X_test.index
    predictions["Actual Revenue Impact"] = y_test
    predictions["Decision Tree Prediction"] = test_pred
    predictions["Error"] = y_test - test_pred
    predictions["Absolute Error"] = predictions["Error"].abs()
    predictions["Error Direction"] = predictions["Error"].map(error_direction)
    predictions = predictions.sort_values("Case ID")

    rules = build_sklearn_decision_tree_rules(model, tree_features, transform_lookup)

    return {
        "model": model,
        "tree_features": tree_features,
        "transform_lookup": transform_lookup,
        "feature_map": feature_map,
        "importance": importance,
        "stats": stats,
        "metrics": metrics,
        "rules": rules,
        "train_predictions": train_predictions.drop(columns=["Absolute Error"]),
        "predictions": predictions.drop(columns=["Absolute Error"]),
        "largest": predictions.nlargest(10, "Absolute Error").drop(columns=["Absolute Error"]),
    }


def xgboost_min_split_loss(y_train):
    root_sse = float(((y_train - y_train.mean()) ** 2).sum())
    return root_sse * TREE_MIN_NODE_GAIN / XGBOOST_TREES


def build_xgboost_tree_rules(model, tree_features, transform_lookup, min_split_loss):
    booster_frame = model.get_booster().trees_to_dataframe()
    rows = []

    for _, row in booster_frame.iterrows():
        tree_feature = row["Feature"]
        original_feature = ""
        original_threshold = ""
        transform = ""
        skewness = ""
        skew_direction = ""

        if tree_feature != "Leaf":
            info = transform_lookup[tree_feature]
            original_feature = info["original"]
            original_threshold = tree_original_threshold(
                tree_feature,
                float(row["Split"]),
                transform_lookup,
            )
            transform = info["transform"]
            skewness = info["skewness"]
            skew_direction = info["skew_direction"]

        gain = float(row["Gain"])
        rows.append(
            {
                "Tree": int(row["Tree"]),
                "Node": int(row["Node"]),
                "Node Type": "Leaf" if tree_feature == "Leaf" else "Split",
                "Original Feature": original_feature,
                "Tree Feature": "" if tree_feature == "Leaf" else tree_feature,
                "Transform": transform,
                "Original Threshold": original_threshold,
                "Native Gain": gain,
                "Minimum Native Gain": "" if tree_feature == "Leaf" else min_split_loss,
                "Meets 5 Percent Stop Rule": ""
                if tree_feature == "Leaf"
                else "Yes"
                if gain >= min_split_loss
                else "No",
                "Cover": float(row["Cover"]),
                "Yes Child": row["Yes"],
                "No Child": row["No"],
                "Missing Child": row["Missing"],
                "Skewness": skewness,
                "Skew Direction": skew_direction,
            }
        )

    return pd.DataFrame(rows)


def build_xgboost_split_audit(xgboost_rules, min_split_loss):
    split_rows = xgboost_rules[xgboost_rules["Node Type"] == "Split"].copy()
    below_threshold = split_rows[
        pd.to_numeric(split_rows["Native Gain"], errors="coerce") < min_split_loss
    ]
    return pd.DataFrame(
        [
            {"Check": "XGBoost Split Rows", "Value": len(split_rows), "Result": "trained ensemble splits"},
            {"Check": "Minimum Native Gain Threshold", "Value": min_split_loss, "Result": "5% root SSE divided by tree count"},
            {"Check": "Splits Below Threshold", "Value": len(below_threshold), "Result": "pass" if below_threshold.empty else "review"},
            {"Check": "Leaf Stop Rule", "Value": "gamma minimum split loss", "Result": "native XGBoost enforcement"},
        ]
    )


def fit_xgboost_model(model_df):
    base_df = model_df[["company_id", "year"] + FEATURES + [TARGET]].dropna().copy()
    raw_X = base_df[FEATURES]
    y = base_df[TARGET]
    raw_X_train, raw_X_test, y_train, y_test = train_test_split(
        raw_X,
        y,
        test_size=0.30,
        random_state=42,
    )
    tree_profile = build_feature_profile(raw_X_train)
    tree_df, tree_features, transform_lookup, feature_map = build_decision_tree_inputs(
        base_df,
        tree_profile,
    )
    X = tree_df[tree_features]
    X_train = X.loc[raw_X_train.index]
    X_test = X.loc[raw_X_test.index]
    min_split_loss = xgboost_min_split_loss(y_train)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=XGBOOST_TREES,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=20,
        gamma=min_split_loss,
        tree_method="hist",
        eval_metric="rmse",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    metrics = build_metrics(y_train, train_pred, y_test, test_pred, len(FEATURES))
    stats = metrics_table(metrics)
    stats.loc[len(stats)] = {
        "Metric": "Model",
        "Training Data": "Skew-aware XGBoost tree ensemble",
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Trees",
        "Training Data": XGBOOST_TREES,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Max Depth",
        "Training Data": XGBOOST_MAX_DEPTH,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Learning Rate",
        "Training Data": XGBOOST_LEARNING_RATE,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Minimum Native Split Gain",
        "Training Data": min_split_loss,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Leaf Stop Rule",
        "Training Data": "gamma = 5% root SSE / tree count",
        "Test Data": "",
    }

    importance = pd.DataFrame(
        {
            "Tree Feature": tree_features,
            "Importance": model.feature_importances_,
        }
    )
    importance["Original Feature"] = importance["Tree Feature"].map(lambda value: transform_lookup[value]["original"])
    importance["Transform"] = importance["Tree Feature"].map(lambda value: transform_lookup[value]["transform"])
    importance["Skewness"] = importance["Tree Feature"].map(lambda value: transform_lookup[value]["skewness"])
    importance["Skew Direction"] = importance["Tree Feature"].map(lambda value: transform_lookup[value]["skew_direction"])
    importance = importance.sort_values("Importance", ascending=False)

    predictions = base_df.loc[X_test.index, FEATURES].copy()
    predictions["Case ID"] = X_test.index
    predictions["Actual Revenue Impact"] = y_test
    predictions["XGBoost Prediction"] = test_pred
    predictions["Error"] = y_test - test_pred
    predictions["Absolute Error"] = predictions["Error"].abs()
    predictions["Error Direction"] = predictions["Error"].map(error_direction)
    predictions["year"] = base_df.loc[X_test.index, "year"]
    predictions = predictions.sort_values("Case ID")
    train_predictions = base_df.loc[X_train.index, FEATURES].copy()
    train_predictions["Case ID"] = X_train.index
    train_predictions["Actual Revenue Impact"] = y_train
    train_predictions["XGBoost Prediction"] = train_pred
    train_predictions["Error"] = y_train - train_pred
    train_predictions["Absolute Error"] = train_predictions["Error"].abs()
    train_predictions["Error Direction"] = train_predictions["Error"].map(error_direction)
    train_predictions["year"] = base_df.loc[X_train.index, "year"]
    train_predictions = train_predictions.sort_values("Case ID")
    rules = build_xgboost_tree_rules(model, tree_features, transform_lookup, min_split_loss)
    split_audit = build_xgboost_split_audit(rules, min_split_loss)

    return {
        "model": model,
        "stats": stats,
        "metrics": metrics,
        "feature_map": feature_map,
        "importance": importance,
        "rules": rules,
        "split_audit": split_audit,
        "train_predictions": train_predictions.drop(columns=["Absolute Error"]),
        "predictions": predictions.drop(columns=["Absolute Error"]),
        "largest": predictions.nlargest(10, "Absolute Error").drop(columns=["Absolute Error"]),
    }


def fit_adoption_decision_tree_model(full_df):
    base_df = full_df[["company_id", "year", ADOPTION_TARGET] + ADOPTION_FEATURES].dropna().copy()
    raw_X = base_df[ADOPTION_FEATURES]
    y = base_df[ADOPTION_TARGET]
    raw_X_train, raw_X_test, y_train, y_test = train_test_split(
        raw_X,
        y,
        test_size=0.30,
        random_state=42,
    )
    feature_profile = build_feature_profile_for_columns(raw_X_train, ADOPTION_FEATURES)
    tree_df, tree_features, transform_lookup, feature_map = build_xgboost_inputs_for_features(
        base_df,
        ADOPTION_FEATURES,
        feature_profile,
    )
    X_train = tree_df.loc[raw_X_train.index, tree_features]
    X_test = tree_df.loc[raw_X_test.index, tree_features]

    model = DecisionTreeRegressor(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = predict_pruned_decision_tree(model, X_train)
    test_pred = predict_pruned_decision_tree(model, X_test)
    metrics = build_metrics(y_train, train_pred, y_test, test_pred, len(tree_features))
    stats = metrics_table(metrics)
    stats.loc[len(stats)] = {
        "Metric": "Model",
        "Training Data": "Supervised adoption-level decision tree",
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Target Column",
        "Training Data": ADOPTION_TARGET,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Max Depth",
        "Training Data": pruned_tree_depth(model),
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Leaves",
        "Training Data": pruned_tree_leaf_count(model),
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Minimum Leaf Rows",
        "Training Data": TREE_MIN_SAMPLES_LEAF,
        "Test Data": "",
    }

    importance = pruned_tree_importance(model, tree_features, transform_lookup)
    rules = build_sklearn_decision_tree_rules(model, tree_features, transform_lookup)

    predictions = base_df.loc[X_test.index, ADOPTION_FEATURES].copy()
    predictions["Case ID"] = X_test.index
    predictions["Actual AI Adoption Level"] = y_test
    predictions["Adoption Tree Prediction"] = test_pred
    predictions["Error"] = y_test - test_pred
    predictions["Absolute Error"] = predictions["Error"].abs()
    predictions["Error Direction"] = predictions["Error"].map(error_direction)
    predictions["year"] = base_df.loc[X_test.index, "year"]
    predictions = predictions.sort_values("Case ID")

    return {
        "model": model,
        "tree_features": tree_features,
        "transform_lookup": transform_lookup,
        "feature_map": feature_map,
        "importance": importance,
        "rules": rules,
        "stats": stats,
        "metrics": metrics,
        "predictions": predictions.drop(columns=["Absolute Error"]),
        "largest": predictions.nlargest(10, "Absolute Error").drop(columns=["Absolute Error"]),
    }


def fit_adoption_xgboost_model(full_df):
    base_df = full_df[["company_id", "year", ADOPTION_TARGET] + ADOPTION_FEATURES].dropna().copy()
    raw_X = base_df[ADOPTION_FEATURES]
    y = base_df[ADOPTION_TARGET]
    raw_X_train, raw_X_test, y_train, y_test = train_test_split(
        raw_X,
        y,
        test_size=0.30,
        random_state=42,
    )
    feature_profile = build_feature_profile_for_columns(raw_X_train, ADOPTION_FEATURES)
    tree_df, tree_features, transform_lookup, feature_map = build_xgboost_inputs_for_features(
        base_df,
        ADOPTION_FEATURES,
        feature_profile,
    )
    X_train = tree_df.loc[raw_X_train.index, tree_features]
    X_test = tree_df.loc[raw_X_test.index, tree_features]
    min_split_loss = xgboost_min_split_loss(y_train)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=XGBOOST_TREES,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=20,
        gamma=min_split_loss,
        tree_method="hist",
        eval_metric="rmse",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    metrics = build_metrics(y_train, train_pred, y_test, test_pred, len(tree_features))
    stats = metrics_table(metrics)
    stats.loc[len(stats)] = {
        "Metric": "Model",
        "Training Data": "Supervised adoption-level XGBoost regressor",
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Target Column",
        "Training Data": ADOPTION_TARGET,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Trees",
        "Training Data": XGBOOST_TREES,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Max Depth",
        "Training Data": XGBOOST_MAX_DEPTH,
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Learning Rate",
        "Training Data": XGBOOST_LEARNING_RATE,
        "Test Data": "",
    }

    importance = pd.DataFrame(
        {
            "Tree Feature": tree_features,
            "Importance": model.feature_importances_,
        }
    )
    importance["Original Feature"] = importance["Tree Feature"].map(lambda value: transform_lookup[value]["original"])
    importance["Transform"] = importance["Tree Feature"].map(lambda value: transform_lookup[value]["transform"])
    importance["Skewness"] = importance["Tree Feature"].map(lambda value: transform_lookup[value]["skewness"])
    importance["Skew Direction"] = importance["Tree Feature"].map(lambda value: transform_lookup[value]["skew_direction"])
    importance = importance.sort_values("Importance", ascending=False)

    predictions = base_df.loc[X_test.index, ADOPTION_FEATURES].copy()
    predictions["Case ID"] = X_test.index
    predictions["Actual AI Adoption Level"] = y_test
    predictions["Adoption XGBoost Prediction"] = test_pred
    predictions["Error"] = y_test - test_pred
    predictions["Absolute Error"] = predictions["Error"].abs()
    predictions["Error Direction"] = predictions["Error"].map(error_direction)
    predictions["year"] = base_df.loc[X_test.index, "year"]
    predictions = predictions.sort_values("Case ID")

    rules = build_xgboost_tree_rules(model, tree_features, transform_lookup, min_split_loss)
    split_audit = build_xgboost_split_audit(rules, min_split_loss)
    split_audit.loc[len(split_audit)] = {
        "Check": "Target Column",
        "Value": ADOPTION_TARGET,
        "Result": "supervised adoption-level target",
    }

    return {
        "model": model,
        "stats": stats,
        "metrics": metrics,
        "feature_map": feature_map,
        "importance": importance,
        "rules": rules,
        "split_audit": split_audit,
        "predictions": predictions.drop(columns=["Absolute Error"]),
        "largest": predictions.nlargest(10, "Absolute Error").drop(columns=["Absolute Error"]),
    }


def build_separated_feature_splits(model_df):
    base_df = model_df[["company_id", "year"] + FEATURES + [TARGET]].dropna().copy()
    raw_X = base_df[FEATURES]
    y = base_df[TARGET]
    raw_X_train, raw_X_test, y_train, y_test = train_test_split(
        raw_X,
        y,
        test_size=0.30,
        random_state=42,
    )
    tree_profile = build_feature_profile(raw_X_train)
    tree_df, tree_features, transform_lookup, feature_map = build_decision_tree_inputs(
        base_df,
        tree_profile,
    )
    rows = []

    for tree_feature in tree_features:
        X = tree_df[[tree_feature]]
        X_train = X.loc[raw_X_train.index]
        X_test = X.loc[raw_X_test.index]
        model = DecisionTreeRegressor(
            max_depth=1,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            random_state=42,
        )
        model.fit(X_train, y_train)
        train_pred = predict_pruned_decision_tree(model, X_train)
        test_pred = predict_pruned_decision_tree(model, X_test)
        tree = model.tree_
        split_exists = tree.children_left[0] != tree.children_right[0]
        info = transform_lookup[tree_feature]

        if split_exists:
            left_child = tree.children_left[0]
            right_child = tree.children_right[0]
            threshold = tree_original_threshold(
                tree_feature,
                float(tree.threshold[0]),
                transform_lookup,
            )
            variance_gain = sklearn_node_variance_gain(model, 0)
            left_prediction = float(tree.value[left_child][0][0])
            right_prediction = float(tree.value[right_child][0][0])
            left_rows = int(tree.n_node_samples[left_child])
            right_rows = int(tree.n_node_samples[right_child])
            status = "Split" if variance_gain > TREE_MIN_NODE_GAIN else "Leaf"
            stop_rule = "" if variance_gain > TREE_MIN_NODE_GAIN else "Node variance gain <= 5%"
        else:
            threshold = ""
            variance_gain = ""
            left_prediction = ""
            right_prediction = ""
            left_rows = ""
            right_rows = ""
            status = "Leaf"
            stop_rule = "No valid split"

        rows.append(
            {
                "Variable": info["original"],
                "Tree Feature": tree_feature,
                "Transform": info["transform"],
                "Status": status,
                "Threshold": threshold,
                "Node Variance Gain": variance_gain,
                "Train R Square": r2_score(y_train, train_pred),
                "Test R Square": r2_score(y_test, test_pred),
                "Left Rows": left_rows,
                "Left Prediction": left_prediction,
                "Right Rows": right_rows,
                "Right Prediction": right_prediction,
                "Stop Rule": stop_rule,
                "Skewness": info["skewness"],
                "Skew Direction": info["skew_direction"],
                "Skew Strength": info["skew_strength"],
            }
        )

    result = pd.DataFrame(rows)
    sort_values = pd.to_numeric(result["Node Variance Gain"], errors="coerce").fillna(-np.inf)
    return (
        result.assign(_sort=sort_values)
        .sort_values("_sort", ascending=False)
        .drop(columns="_sort"),
        feature_map,
    )


def render_separated_feature_splits(separated_splits):
    cards = []
    for _, row in separated_splits.iterrows():
        variance_gain = pd.to_numeric(row["Node Variance Gain"], errors="coerce")
        variance_gain_label = "" if pd.isna(variance_gain) else f"{float(variance_gain) * 100:.2f}%"
        if row["Status"] == "Split":
            branch_html = f"""
                <div class="split-branches">
                    <span><b>Left</b>{html.escape(row["Variable"])} <= {number(row["Threshold"])}<br>{money(row["Left Prediction"])} / {int(row["Left Rows"]):,} rows</span>
                    <span><b>Right</b>{html.escape(row["Variable"])} > {number(row["Threshold"])}<br>{money(row["Right Prediction"])} / {int(row["Right Rows"]):,} rows</span>
                </div>
            """
        else:
            branch_html = f'<div class="split-branches"><span><b>Leaf</b>{html.escape(str(row["Stop Rule"]))}</span></div>'

        cards.append(
            f"""
            <div class="split-card">
                <div class="split-card-head">
                    <strong>{html.escape(row["Variable"])}</strong>
                    <span>{html.escape(row["Transform"])} / skew {float(row["Skewness"]):.3f}</span>
                </div>
                <div class="split-metrics">
                    <span><b>Node gain</b>{variance_gain_label}</span>
                    <span><b>Test R2</b>{float(row["Test R Square"]):.4f}</span>
                </div>
                {branch_html}
            </div>
            """
        )

    return f"""
    <div class="calculated-chart">
        <h3>Separated Variable Split Checks</h3>
        <p>Each feature is evaluated independently. No feature is nested under another feature.</p>
        <div class="split-card-grid">{''.join(cards)}</div>
    </div>
    """


def build_tree_model_selection(main_metrics, decision_tree_metrics, xgboost_metrics, stacked_metrics=None):
    candidates = [
        ("Linear Regression", main_metrics),
        ("Single Decision Tree", decision_tree_metrics),
        ("XGBoost Tree Ensemble", xgboost_metrics),
    ]
    if stacked_metrics is not None:
        candidates.append(("Stacked Ensemble Regression", stacked_metrics))
    selected_model, selected_metrics = max(
        candidates,
        key=lambda item: item[1]["test_r2"],
    )
    xgboost_gain = xgboost_metrics["test_r2"] - main_metrics["test_r2"]
    stacked_gain = None if stacked_metrics is None else stacked_metrics["test_r2"] - xgboost_metrics["test_r2"]

    selection_rows = [
        {
            "Check": "Linear Regression Test R Square",
            "Value": main_metrics["test_r2"],
            "Result": "baseline",
        },
        {
            "Check": "Single Decision Tree Test R Square",
            "Value": decision_tree_metrics["test_r2"],
            "Result": "diagnostic candidate",
        },
        {
            "Check": "XGBoost Test R Square",
            "Value": xgboost_metrics["test_r2"],
            "Result": "tree ensemble candidate",
        },
        {
            "Check": "XGBoost R Square Gain",
            "Value": xgboost_gain,
            "Result": "improved" if xgboost_gain > 0 else "did not improve",
        },
    ]
    if stacked_metrics is not None:
        selection_rows.extend(
            [
                {
                    "Check": "Stacked Ensemble Regression Test R Square",
                    "Value": stacked_metrics["test_r2"],
                    "Result": "meta-regression candidate",
                },
                {
                    "Check": "Stacked Ensemble R Square Gain",
                    "Value": stacked_gain,
                    "Result": "improved over XGBoost" if stacked_gain > 0 else "did not improve over XGBoost",
                },
            ]
        )
    selection_rows.append(
        {
            "Check": "Final Selected Model",
            "Value": selected_model,
            "Result": "highest held-out test R Square",
        }
    )
    selection = pd.DataFrame(selection_rows)

    final_stats = metrics_table(selected_metrics)
    final_stats.insert(0, "Selected Model", selected_model)
    return selection, final_stats
