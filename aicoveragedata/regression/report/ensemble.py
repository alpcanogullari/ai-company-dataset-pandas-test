import html

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .config import (
    FEATURES,
    TARGET,
    TREE_MAX_DEPTH,
    TREE_MIN_SAMPLES_LEAF,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_TREES,
)
from .formatting import adjusted_r2, error_direction, money, number, scale_value
from .modeling import metrics_table
from .profiles import build_feature_profile
from .skew_transform import apply_skew_transform, build_skew_transform_plan
from .tree_models import build_decision_tree_inputs, predict_pruned_decision_tree, xgboost_min_split_loss

STACKING_FOLDS = 5


def render_stacked_xgboost_visual(stacked_model):
    predictions = stacked_model["predictions"].copy()
    metrics = stacked_model["metrics"]
    model = stacked_model["model"]
    signal_columns = stacked_model["signal_columns"]
    signal = predictions["Stacked Regression Prediction"].astype(float)
    actual = predictions["Actual Revenue Impact"].astype(float)
    errors = predictions["Error"].astype(float)

    sampled = predictions.sample(min(len(predictions), 3000), random_state=42)
    signal_low = float(signal.quantile(0.01))
    signal_high = float(signal.quantile(0.99))
    actual_low = float(actual.quantile(0.01))
    actual_high = float(actual.quantile(0.99))
    if signal_high <= signal_low:
        signal_high = signal_low + 1
    if actual_high <= actual_low:
        actual_high = actual_low + 1

    width = 820
    height = 470
    left = 82
    right = 34
    top = 34
    bottom = 74
    plot_width = width - left - right
    plot_height = height - top - bottom

    grid = []
    labels = []
    for index in range(5):
        x_value = signal_low + (signal_high - signal_low) * index / 4
        y_value = actual_low + (actual_high - actual_low) * index / 4
        x = scale_value(x_value, signal_low, signal_high, left, left + plot_width)
        y = scale_value(y_value, actual_low, actual_high, top + plot_height, top)
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" class="svg-grid"></line>')
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="svg-grid"></line>')
        labels.append(f'<text x="{x:.2f}" y="{top + plot_height + 22}" class="svg-tick" text-anchor="middle">{money(x_value)}</text>')
        labels.append(f'<text x="{left - 10}" y="{y + 4:.2f}" class="svg-tick" text-anchor="end">{money(y_value)}</text>')

    points = []
    for x_value, y_value in sampled[["Stacked Regression Prediction", "Actual Revenue Impact"]].itertuples(index=False, name=None):
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            continue
        x = scale_value(x_value, signal_low, signal_high, left, left + plot_width)
        y = scale_value(y_value, actual_low, actual_high, top + plot_height, top)
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" class="stacked-point"></circle>')

    line_low = max(signal_low, actual_low)
    line_high = min(signal_high, actual_high)
    if line_high <= line_low:
        line_low = min(signal_low, actual_low)
        line_high = max(signal_high, actual_high)
    x1 = scale_value(line_low, signal_low, signal_high, left, left + plot_width)
    y1 = scale_value(line_low, actual_low, actual_high, top + plot_height, top)
    x2 = scale_value(line_high, signal_low, signal_high, left, left + plot_width)
    y2 = scale_value(line_high, actual_low, actual_high, top + plot_height, top)
    line = f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="stacked-regression-line"></line>'

    error_low = float(errors.quantile(0.01))
    error_high = float(errors.quantile(0.99))
    if error_high <= error_low:
        error_high = error_low + 1
    clipped_errors = errors.clip(lower=error_low, upper=error_high)
    counts, edges = np.histogram(clipped_errors, bins=28, range=(error_low, error_high))
    max_count = max(int(counts.max()), 1)
    hist_width = 560
    hist_height = 230
    hist_left = 42
    hist_bottom = 36
    hist_plot_height = hist_height - hist_bottom - 16
    bar_width = (hist_width - hist_left - 20) / len(counts)
    bars = []
    zero_x = scale_value(0, error_low, error_high, hist_left, hist_width - 20)
    for index, count in enumerate(counts):
        x = hist_left + index * bar_width
        h = count / max_count * hist_plot_height
        y = 16 + hist_plot_height - h
        center = (edges[index] + edges[index + 1]) / 2
        css_class = "residual-bar over" if center < 0 else "residual-bar under"
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 1, 1):.2f}" height="{h:.2f}" class="{css_class}"></rect>')

    terms = " + ".join(
        f"{number(coefficient)} x {feature}"
        for feature, coefficient in zip(signal_columns, model.coef_)
    )
    formula = f"revenue_impact = {money(model.intercept_)} + {terms}"
    card_rows = [
        ("Test R Square", f"{metrics['test_r2']:.4f}"),
        ("Test RMSE", money(metrics["test_rmse"])),
        ("Test MAE", money(metrics["test_mae"])),
        ("Test Rows", f"{metrics['test_rows']:,}"),
    ]
    cards = "".join(
        f'<div class="stacked-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in card_rows
    )

    return f"""
    <div class="stacked-hero">
        <div>
            <h3>Stacked Ensemble Regression</h3>
            <p>Linear regression, decision tree, and XGBoost act as base learners. The meta-model learns the final weighted combination.</p>
            <div class="stacked-formula">{html.escape(formula)}</div>
        </div>
        <div class="stacked-card-grid">{cards}</div>
    </div>
    <div class="stacked-visual-grid">
        <div class="calculated-chart">
            <h3>Ensemble Prediction vs Revenue Impact</h3>
            <svg class="calculated-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Ensemble prediction versus revenue impact">
                <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="svg-plot-bg"></rect>
                {''.join(grid)}
                <g>{''.join(points)}</g>
                {line}
                <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="svg-axis"></line>
                <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="svg-axis"></line>
                {''.join(labels)}
                <text x="{left + plot_width / 2}" y="{height - 18}" class="svg-axis-label" text-anchor="middle">Stacked Ensemble Prediction</text>
                <text x="18" y="{top + plot_height / 2}" class="svg-axis-label" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})">Actual Revenue Impact</text>
            </svg>
            <div class="chart-caption">Sampled {len(sampled):,} of {len(predictions):,} held-out test rows. The bright line is actual = prediction.</div>
        </div>
        <div class="calculated-chart">
            <h3>Residual Distribution</h3>
            <svg class="calculated-svg stacked-histogram" viewBox="0 0 {hist_width} {hist_height}" role="img" aria-label="Stacked model residual distribution">
                <rect x="{hist_left}" y="16" width="{hist_width - hist_left - 20}" height="{hist_plot_height}" class="svg-plot-bg"></rect>
                <line x1="{zero_x:.2f}" y1="16" x2="{zero_x:.2f}" y2="{16 + hist_plot_height}" class="svg-diagonal"></line>
                {''.join(bars)}
                <line x1="{hist_left}" y1="{16 + hist_plot_height}" x2="{hist_width - 20}" y2="{16 + hist_plot_height}" class="svg-axis"></line>
                <text x="{hist_left}" y="{hist_height - 12}" class="svg-tick" text-anchor="start">{money(error_low)}</text>
                <text x="{hist_width - 20}" y="{hist_height - 12}" class="svg-tick" text-anchor="end">{money(error_high)}</text>
                <text x="{hist_width / 2}" y="{hist_height - 12}" class="svg-axis-label" text-anchor="middle">Prediction Error</text>
            </svg>
            <div class="stacked-legend">
                <span><i class="under-error-key"></i>Underpredicted</span>
                <span><i class="over-error-key"></i>Overpredicted</span>
            </div>
            <div class="chart-caption">Errors near zero mean the calibrated ensemble prediction is close to actual revenue impact.</div>
        </div>
    </div>
    """


def combine_prediction_signals(linear_predictions, decision_tree_predictions, xgboost_predictions):
    linear = linear_predictions[["Case ID", "Actual Revenue Impact", "Linear Regression Prediction"]].copy()
    tree = decision_tree_predictions[["Case ID", "Decision Tree Prediction"]].copy()
    xgboost = xgboost_predictions[["Case ID", "XGBoost Prediction"]].copy()
    combined = linear.merge(tree, on="Case ID", how="inner").merge(xgboost, on="Case ID", how="inner")
    return combined.sort_values("Case ID").reset_index(drop=True)


def build_out_of_fold_base_predictions(model_df, train_case_ids):
    base_df = model_df[["company_id", "year"] + FEATURES + [TARGET]].dropna().copy()
    train_case_ids = np.array(sorted(train_case_ids))
    oof = pd.DataFrame(
        {
            "Case ID": train_case_ids,
            "Actual Revenue Impact": base_df.loc[train_case_ids, TARGET].to_numpy(),
        }
    )
    for column in [
        "Linear Regression Prediction",
        "Decision Tree Prediction",
        "XGBoost Prediction",
    ]:
        oof[column] = np.nan

    folds = KFold(n_splits=STACKING_FOLDS, shuffle=True, random_state=42)
    for fold_train_positions, fold_valid_positions in folds.split(train_case_ids):
        fold_train_ids = train_case_ids[fold_train_positions]
        fold_valid_ids = train_case_ids[fold_valid_positions]
        X_train = base_df.loc[fold_train_ids, FEATURES]
        X_valid = base_df.loc[fold_valid_ids, FEATURES]
        y_train = base_df.loc[fold_train_ids, TARGET]

        skew_plan = build_skew_transform_plan(X_train)
        linear_X_train = apply_skew_transform(X_train, skew_plan)
        linear_X_valid = apply_skew_transform(X_valid, skew_plan)
        linear_model = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("model", LinearRegression()),
            ]
        )
        linear_model.fit(linear_X_train, y_train)
        oof.loc[oof["Case ID"].isin(fold_valid_ids), "Linear Regression Prediction"] = linear_model.predict(linear_X_valid)

        tree_profile = build_feature_profile(base_df.loc[fold_train_ids])
        tree_df, tree_features, _transform_lookup, _feature_map = build_decision_tree_inputs(
            base_df,
            tree_profile,
        )
        tree_model = DecisionTreeRegressor(
            max_depth=TREE_MAX_DEPTH,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            random_state=42,
        )
        tree_model.fit(tree_df.loc[fold_train_ids, tree_features], y_train)
        oof.loc[oof["Case ID"].isin(fold_valid_ids), "Decision Tree Prediction"] = predict_pruned_decision_tree(
            tree_model,
            tree_df.loc[fold_valid_ids, tree_features],
        )

        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=XGBOOST_TREES,
            max_depth=XGBOOST_MAX_DEPTH,
            learning_rate=XGBOOST_LEARNING_RATE,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            min_child_weight=20,
            gamma=xgboost_min_split_loss(y_train),
            tree_method="hist",
            eval_metric="rmse",
            random_state=42,
            n_jobs=4,
        )
        xgb_model.fit(tree_df.loc[fold_train_ids, tree_features], y_train, verbose=False)
        oof.loc[oof["Case ID"].isin(fold_valid_ids), "XGBoost Prediction"] = xgb_model.predict(
            tree_df.loc[fold_valid_ids, tree_features]
        )

    if oof[["Linear Regression Prediction", "Decision Tree Prediction", "XGBoost Prediction"]].isna().any().any():
        raise ValueError("Out-of-fold ensemble predictions contain missing values.")
    return oof.sort_values("Case ID").reset_index(drop=True)


def build_stacked_ensemble_regression(linear_predictions, decision_tree_model, xgboost_model, model_df):
    """Combine multiple base learners into one stacked regression."""
    train_case_ids = linear_predictions["train_predictions"]["Case ID"].to_numpy()
    train_predictions = build_out_of_fold_base_predictions(
        model_df,
        train_case_ids,
    )
    predictions = combine_prediction_signals(
        linear_predictions["predictions"],
        decision_tree_model["predictions"],
        xgboost_model["predictions"],
    )
    signal_columns = [
        "Linear Regression Prediction",
        "Decision Tree Prediction",
        "XGBoost Prediction",
    ]

    X_train = train_predictions[signal_columns]
    y_train = train_predictions["Actual Revenue Impact"]
    X_test = predictions[signal_columns]
    y_test = predictions["Actual Revenue Impact"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    train_stacked_prediction = model.predict(X_train)
    stacked_prediction = model.predict(X_test)

    predictions["Stacked Regression Prediction"] = stacked_prediction
    predictions["Error"] = y_test - stacked_prediction
    predictions["Absolute Error"] = predictions["Error"].abs()
    predictions["Error Direction"] = predictions["Error"].map(error_direction)

    train_r2 = r2_score(y_train, train_stacked_prediction)
    test_r2 = r2_score(y_test, stacked_prediction)
    metrics = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_adj_r2": adjusted_r2(train_r2, len(train_predictions), len(signal_columns)),
        "test_adj_r2": adjusted_r2(test_r2, len(predictions), len(signal_columns)),
        "train_mae": mean_absolute_error(y_train, train_stacked_prediction),
        "test_mae": mean_absolute_error(y_test, stacked_prediction),
        "train_rmse": mean_squared_error(y_train, train_stacked_prediction) ** 0.5,
        "test_rmse": mean_squared_error(y_test, stacked_prediction) ** 0.5,
        "train_rows": len(train_predictions),
        "test_rows": len(predictions),
    }
    stats = metrics_table(metrics)
    stats.loc[len(stats)] = {
        "Metric": "Model",
        "Training Data": "Stacked regression on linear, decision tree, and XGBoost base learners",
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Base Learners",
        "Training Data": "Linear regression, decision tree, XGBoost",
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Independent Variables",
        "Training Data": ", ".join(signal_columns),
        "Test Data": "",
    }
    stats.loc[len(stats)] = {
        "Metric": "Dependent Variable",
        "Training Data": "revenue_impact",
        "Test Data": "",
    }

    coefficients = pd.DataFrame(
        [{"Feature": "Intercept", "Coefficient": model.intercept_}]
        + [
            {"Feature": feature, "Coefficient": coefficient}
            for feature, coefficient in zip(signal_columns, model.coef_)
        ]
    )
    coefficients["Abs Coefficient"] = coefficients["Coefficient"].abs()
    coefficients["Direction"] = np.where(coefficients["Coefficient"] >= 0, "Positive", "Negative")

    audit = pd.DataFrame(
        [
            {"Check": "Base learners", "Value": "Linear regression, decision tree, XGBoost", "Result": "combined into one ensemble"},
            {"Check": "Base learner signals", "Value": ", ".join(signal_columns), "Result": "inputs to the meta-model"},
            {"Check": "Stacking method", "Value": f"{STACKING_FOLDS}-fold out-of-fold predictions", "Result": "prevents in-sample meta-training"},
            {"Check": "Meta model", "Value": "LinearRegression", "Result": "learns final weighted combination"},
            {"Check": "Independent variables", "Value": ", ".join(signal_columns), "Result": "model prediction signals"},
            {"Check": "Dependent variable", "Value": "revenue_impact", "Result": "y-axis / target"},
            {"Check": "Training rows", "Value": len(train_predictions), "Result": "out-of-fold predictions from shared 70% training rows"},
            {"Check": "Test rows", "Value": len(predictions), "Result": "shared 30% held-out rows"},
            {"Check": "Intercept", "Value": model.intercept_, "Result": "calibration offset"},
        ]
    )

    return {
        "model": model,
        "stats": stats,
        "metrics": metrics,
        "coefficients": coefficients,
        "audit": audit,
        "visual": render_stacked_xgboost_visual(
            {
                "model": model,
                "metrics": metrics,
                "predictions": predictions,
                "signal_columns": signal_columns,
            }
        ),
        "predictions": predictions,
        "largest": predictions.nlargest(10, "Absolute Error").drop(columns=["Absolute Error"]),
    }


def build_stacked_xgboost_regression(linear_predictions, decision_tree_model, xgboost_model, model_df):
    return build_stacked_ensemble_regression(linear_predictions, decision_tree_model, xgboost_model, model_df)
