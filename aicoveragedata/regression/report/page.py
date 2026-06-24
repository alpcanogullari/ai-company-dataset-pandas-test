import html

import pandas as pd

from aicoveragedata.app.agent_widget import (
    agent_widget_markup,
    agent_widget_script,
    agent_widget_styles,
)
from .config import (
    DATA_PATH,
    DOWNLOAD_DIR,
    FEATURES,
    LAG_DECAY_FEATURES,
    NEAR_ZERO_CORRELATION_LIMIT,
    OUTPUT_PATH,
    TARGET,
)
from .formatting import add_error_percentages, money, number, table_html
from .ensemble import build_stacked_xgboost_regression
from .lag_analysis import (
    build_lag_decay_data,
    build_time_lag_data,
    fit_current_plus_lag_model,
    fit_lagged_model,
    render_lag_decay_chart,
    render_lag_heatmap,
    render_time_lag_detail,
)
from .modeling import fit_model, metrics_table, pipeline_audit_table
from .nonlinearity import (
    build_near_zero_pairwise_nonlinearity_checks,
    build_nonlinear_correlation_checks,
    nonlinear_correlation_note,
)
from .profiles import build_variable_skewness_table
from .tables import export_tables, format_table
from .tree_models import (
    build_separated_feature_splits,
    build_tree_model_selection,
    fit_adoption_decision_tree_model,
    fit_adoption_xgboost_model,
    fit_decision_tree,
    fit_xgboost_model,
    render_decision_tree,
    render_feature_range_overview,
    render_separated_feature_tree_visual,
    render_separated_feature_splits,
    render_xgboost_ensemble_visual,
)
from .visuals import outlier_cards, skewness_distribution_cards, render_actual_vs_predicted

REGRESSION_THEME_CSS = """
        :root {
            color-scheme: dark;
            --page-bg: #05070c;
            --surface: rgba(12, 18, 30, 0.94);
            --surface-strong: rgba(18, 28, 45, 0.96);
            --surface-soft: rgba(8, 13, 22, 0.88);
            --line: rgba(96, 220, 255, 0.24);
            --line-strong: rgba(96, 220, 255, 0.38);
            --text: #e8f7ff;
            --muted: #91a9bd;
            --accent: #36d8ff;
            --accent-hot: #ff65c8;
            --accent-soft: rgba(54, 216, 255, 0.16);
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            background:
                repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.035) 0 1px, transparent 1px 96px),
                repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.025) 0 1px, transparent 1px 96px),
                linear-gradient(135deg, #05070c 0%, #101927 52%, #06080f 100%);
            color: var(--text);
            line-height: 1.45;
        }
        header {
            background: linear-gradient(180deg, rgba(15, 24, 38, 0.98), rgba(6, 9, 16, 0.96));
            border-bottom: 1px solid var(--line);
            box-shadow: 0 18px 48px rgba(54, 216, 255, 0.08);
        }
        h1 {
            letter-spacing: 0;
            text-shadow: 0 0 18px rgba(54, 216, 255, 0.38);
        }
        p, .summary-card span, .split-card-head span, .split-metrics b,
        .split-branches b, .xgb-importance-row em, .xgb-feature-legend,
        .chart-caption, .graph-note, .note, .outlier-vars,
        .skew-card-head span, .skew-axis, .skew-legend, .percentile-strip b {
            color: var(--muted);
        }
        a {
            color: var(--accent);
        }
        .summary-card, .panel, .calculated-chart, .split-card, .xgb-visual-panel,
        .outlier-item, .skew-card, .stacked-hero > div, .stacked-card {
            background: linear-gradient(180deg, var(--surface-strong), var(--surface));
            border-color: var(--line);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.30), 0 0 24px rgba(54, 216, 255, 0.07);
        }
        .toolbar button, .toolbar a {
            background: var(--surface);
            border-color: var(--line-strong);
            color: var(--text);
            box-shadow: inset 0 0 14px rgba(54, 216, 255, 0.07);
            transition: transform 180ms ease, background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }
        .toolbar button:hover, .toolbar a:hover {
            background: var(--surface-strong);
            border-color: var(--accent);
            box-shadow: 0 0 18px rgba(54, 216, 255, 0.14), inset 0 0 18px rgba(54, 216, 255, 0.08);
            transform: translateY(-1px);
        }
        .toolbar button.active {
            background: linear-gradient(135deg, rgba(54, 216, 255, 0.24), rgba(255, 101, 200, 0.18));
            border-color: var(--accent);
            color: #ffffff;
            box-shadow: 0 0 24px rgba(54, 216, 255, 0.22), inset 0 0 20px rgba(255, 101, 200, 0.10);
        }
        .formula, .excel-table, .prediction-image, .skew-svg, .stacked-formula,
        .split-metrics span, .split-branches span, .percentile-strip span {
            background: var(--surface-soft);
            border-color: var(--line);
            color: var(--text);
        }
        .excel-table th, .excel-table td {
            border-color: var(--line);
        }
        .excel-table th {
            background: rgba(54, 216, 255, 0.10);
            color: var(--text);
        }
        .sheet.active {
            animation: regressionSheetIn 260ms ease both;
        }
        .model-score-track, .xgb-importance-track {
            background: #060a12;
            border-color: var(--line);
        }
        .svg-plot-bg {
            fill: #07101a;
            stroke: var(--line);
        }
        .svg-grid {
            stroke: var(--line);
        }
        .svg-axis, .svg-diagonal {
            stroke: #e8f7ff;
        }
        .svg-tick {
            fill: var(--muted);
        }
        .svg-axis-label, .svg-cell-label {
            fill: var(--text);
        }
        .toolbar button:focus-visible, .toolbar a:focus-visible {
            outline: 3px solid rgba(54, 216, 255, 0.25);
            outline-offset: 2px;
        }
        @keyframes regressionSheetIn {
            from {
                opacity: 0;
                transform: translateY(10px) scale(0.99);
                filter: blur(2px);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
                filter: blur(0);
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .toolbar button, .toolbar a, .sheet.active {
                animation: none;
                transition: none;
            }
        }
"""

def write_page():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    obsolete_outputs = [
        "regression_decision_tree_model_selection.csv",
    ]
    for filename in obsolete_outputs:
        stale_file = DOWNLOAD_DIR / filename
        if stale_file.exists():
            stale_file.unlink()

    (
        pipeline,
        model_df,
        linear_train_predictions,
        predictions,
        coefficients,
        anova,
        metrics,
        score_audit,
        skew_transform_audit,
    ) = fit_model()
    full_df = pd.read_csv(DATA_PATH)
    model_audit = pipeline_audit_table(
        pipeline,
        FEATURES,
        metrics,
        "random 70/30 split, random_state=42",
    )
    yearly_lag, overall_lag, lag_summary = build_time_lag_data(model_df)
    lag_decay = build_lag_decay_data(model_df)
    nonlinear_checks = build_nonlinear_correlation_checks(full_df)
    near_zero_pairwise_checks = build_near_zero_pairwise_nonlinearity_checks(full_df)
    lag_model = fit_lagged_model(model_df)
    current_plus_lag_model = fit_current_plus_lag_model(model_df)
    yearly_lag.to_csv(DOWNLOAD_DIR / "regression_time_lag_by_year.csv", index=False)
    overall_lag.to_csv(DOWNLOAD_DIR / "regression_time_lag_correlations.csv", index=False)
    lag_summary.to_csv(DOWNLOAD_DIR / "regression_time_lag_summary.csv", index=False)
    lag_decay.to_csv(DOWNLOAD_DIR / "regression_lag_correlation_decay.csv", index=False)
    nonlinear_checks.to_csv(DOWNLOAD_DIR / "regression_nonlinear_correlation_checks.csv", index=False)
    near_zero_pairwise_checks.to_csv(
        DOWNLOAD_DIR / "regression_near_zero_pairwise_nonlinearity_checks.csv",
        index=False,
    )
    lag_decay[
        (lag_decay["variable"].isin(LAG_DECAY_FEATURES))
        & (lag_decay["lag_years"] <= 4)
    ].to_csv(DOWNLOAD_DIR / "regression_lag_strength_heatmap.csv", index=False)
    current_plus_lag_model["stats"].to_csv(DOWNLOAD_DIR / "regression_current_plus_lag_statistics.csv", index=False)
    current_plus_lag_model["audit"].to_csv(DOWNLOAD_DIR / "regression_current_plus_lag_audit.csv", index=False)
    current_plus_lag_model["coefficients"].to_csv(DOWNLOAD_DIR / "regression_current_plus_lag_coefficients.csv", index=False)
    current_plus_lag_model["components"].to_csv(DOWNLOAD_DIR / "regression_lag_components_by_variable.csv", index=False)
    current_plus_lag_model["predictions"].head(500).to_csv(DOWNLOAD_DIR / "regression_current_plus_lag_predictions_sample.csv", index=False)
    current_plus_lag_model["largest"].to_csv(DOWNLOAD_DIR / "regression_current_plus_lag_largest_error_cases.csv", index=False)
    lag_model["stats"].to_csv(DOWNLOAD_DIR / "regression_lag_model_statistics.csv", index=False)
    lag_model["audit"].to_csv(DOWNLOAD_DIR / "regression_lag_model_audit.csv", index=False)
    lag_model["coefficients"].to_csv(DOWNLOAD_DIR / "regression_lag_model_coefficients.csv", index=False)
    lag_model["predictions"].head(500).to_csv(DOWNLOAD_DIR / "regression_lag_model_predictions_sample.csv", index=False)
    lag_model["largest"].to_csv(DOWNLOAD_DIR / "regression_lag_model_largest_error_cases.csv", index=False)
    export_predictions, largest, top_under, top_over, error_percentages = export_tables(
        predictions,
        coefficients,
    )

    stats = metrics_table(metrics)
    model_df.to_csv(DOWNLOAD_DIR / "regression_skew_transformed_model_data.csv", index=False)
    stats.to_csv(DOWNLOAD_DIR / "regression_statistics.csv", index=False)
    score_audit.to_csv(DOWNLOAD_DIR / "regression_score_audit.csv", index=False)
    skew_transform_audit.to_csv(DOWNLOAD_DIR / "regression_skew_transform_audit.csv", index=False)
    model_audit.to_csv(DOWNLOAD_DIR / "regression_model_audit.csv", index=False)
    anova.to_csv(DOWNLOAD_DIR / "regression_anova.csv", index=False)
    skewness = build_variable_skewness_table(predictions)
    skewness.to_csv(DOWNLOAD_DIR / "regression_variable_skewness.csv", index=False)
    feature_percentiles = skewness.loc[
        skewness["Scope"] == "Test feature",
        [
            "Variable",
            "Rows",
            "P1",
            "P10",
            "P25",
            "Mean",
            "P50",
            "P75",
            "P90",
            "P99",
            "Skewness",
            "Skew Direction",
            "Skew Strength",
        ],
    ].copy()
    feature_percentiles.to_csv(
        DOWNLOAD_DIR / "regression_feature_percentile_ranges.csv",
        index=False,
    )
    decision_tree = fit_decision_tree(model_df)
    separated_feature_splits, _separated_feature_map = build_separated_feature_splits(model_df)
    xgboost_model = fit_xgboost_model(model_df)
    adoption_tree_model = fit_adoption_decision_tree_model(full_df)
    adoption_xgboost_model = fit_adoption_xgboost_model(full_df)
    stacked_xgboost = build_stacked_xgboost_regression(
        {
            "train_predictions": linear_train_predictions,
            "predictions": predictions,
        },
        decision_tree,
        xgboost_model,
        model_df,
    )
    tree_model_selection, final_selected_stats = build_tree_model_selection(
        metrics,
        decision_tree["metrics"],
        xgboost_model["metrics"],
        stacked_xgboost["metrics"],
    )
    final_model_name = final_selected_stats["Selected Model"].iloc[0]
    final_metric_lookup = final_selected_stats.set_index("Metric")
    decision_tree["stats"].to_csv(DOWNLOAD_DIR / "regression_decision_tree_statistics.csv", index=False)
    xgboost_model["stats"].to_csv(DOWNLOAD_DIR / "regression_xgboost_statistics.csv", index=False)
    xgboost_model["feature_map"].to_csv(DOWNLOAD_DIR / "regression_xgboost_feature_map.csv", index=False)
    xgboost_model["importance"].to_csv(DOWNLOAD_DIR / "regression_xgboost_importance.csv", index=False)
    xgboost_model["rules"].to_csv(DOWNLOAD_DIR / "regression_xgboost_tree_rules.csv", index=False)
    xgboost_model["split_audit"].to_csv(DOWNLOAD_DIR / "regression_xgboost_split_audit.csv", index=False)
    xgboost_model["predictions"].head(500).to_csv(DOWNLOAD_DIR / "regression_xgboost_predictions_sample.csv", index=False)
    xgboost_model["largest"].to_csv(DOWNLOAD_DIR / "regression_xgboost_largest_error_cases.csv", index=False)
    adoption_xgboost_model["stats"].to_csv(DOWNLOAD_DIR / "regression_adoption_xgboost_statistics.csv", index=False)
    adoption_xgboost_model["feature_map"].to_csv(DOWNLOAD_DIR / "regression_adoption_xgboost_feature_map.csv", index=False)
    adoption_xgboost_model["importance"].to_csv(DOWNLOAD_DIR / "regression_adoption_xgboost_importance.csv", index=False)
    adoption_xgboost_model["rules"].to_csv(DOWNLOAD_DIR / "regression_adoption_xgboost_tree_rules.csv", index=False)
    adoption_xgboost_model["split_audit"].to_csv(DOWNLOAD_DIR / "regression_adoption_xgboost_split_audit.csv", index=False)
    adoption_xgboost_model["predictions"].head(500).to_csv(DOWNLOAD_DIR / "regression_adoption_xgboost_predictions_sample.csv", index=False)
    adoption_xgboost_model["largest"].to_csv(DOWNLOAD_DIR / "regression_adoption_xgboost_largest_error_cases.csv", index=False)
    adoption_tree_model["stats"].to_csv(DOWNLOAD_DIR / "regression_adoption_tree_statistics.csv", index=False)
    adoption_tree_model["feature_map"].to_csv(DOWNLOAD_DIR / "regression_adoption_tree_feature_map.csv", index=False)
    adoption_tree_model["importance"].to_csv(DOWNLOAD_DIR / "regression_adoption_tree_importance.csv", index=False)
    adoption_tree_model["rules"].to_csv(DOWNLOAD_DIR / "regression_adoption_tree_rules.csv", index=False)
    adoption_tree_model["predictions"].head(500).to_csv(DOWNLOAD_DIR / "regression_adoption_tree_predictions_sample.csv", index=False)
    adoption_tree_model["largest"].to_csv(DOWNLOAD_DIR / "regression_adoption_tree_largest_error_cases.csv", index=False)
    stacked_xgboost["stats"].to_csv(DOWNLOAD_DIR / "regression_stacked_xgboost_statistics.csv", index=False)
    stacked_xgboost["audit"].to_csv(DOWNLOAD_DIR / "regression_stacked_xgboost_audit.csv", index=False)
    stacked_xgboost["coefficients"].to_csv(DOWNLOAD_DIR / "regression_stacked_xgboost_coefficients.csv", index=False)
    stacked_xgboost["predictions"].head(500).to_csv(DOWNLOAD_DIR / "regression_stacked_xgboost_predictions_sample.csv", index=False)
    stacked_xgboost["largest"].to_csv(DOWNLOAD_DIR / "regression_stacked_xgboost_largest_error_cases.csv", index=False)
    stacked_xgboost["stats"].to_csv(DOWNLOAD_DIR / "regression_stacked_ensemble_statistics.csv", index=False)
    stacked_xgboost["audit"].to_csv(DOWNLOAD_DIR / "regression_stacked_ensemble_audit.csv", index=False)
    stacked_xgboost["coefficients"].to_csv(DOWNLOAD_DIR / "regression_stacked_ensemble_coefficients.csv", index=False)
    stacked_xgboost["predictions"].head(500).to_csv(DOWNLOAD_DIR / "regression_stacked_ensemble_predictions_sample.csv", index=False)
    stacked_xgboost["largest"].to_csv(DOWNLOAD_DIR / "regression_stacked_ensemble_largest_error_cases.csv", index=False)
    tree_model_selection.to_csv(DOWNLOAD_DIR / "regression_tree_model_selection.csv", index=False)
    final_selected_stats.to_csv(DOWNLOAD_DIR / "regression_final_selected_model_statistics.csv", index=False)
    decision_tree["feature_map"].to_csv(DOWNLOAD_DIR / "regression_decision_tree_feature_map.csv", index=False)
    decision_tree["importance"].to_csv(DOWNLOAD_DIR / "regression_decision_tree_importance.csv", index=False)
    decision_tree["rules"].to_csv(DOWNLOAD_DIR / "regression_decision_tree_rules.csv", index=False)
    separated_feature_splits.to_csv(DOWNLOAD_DIR / "regression_separated_feature_splits.csv", index=False)
    decision_tree["predictions"].head(500).to_csv(DOWNLOAD_DIR / "regression_decision_tree_predictions_sample.csv", index=False)
    decision_tree["largest"].to_csv(DOWNLOAD_DIR / "regression_decision_tree_largest_error_cases.csv", index=False)

    coefficient_terms = []
    for _, row in coefficients[coefficients["Feature"] != "Intercept"].iterrows():
        sign = "+" if row["Coefficient"] >= 0 else "-"
        coefficient_terms.append(f"{sign} {money(abs(row['Coefficient']))} x {html.escape(row['Feature'])}")
    intercept = coefficients.loc[coefficients["Feature"] == "Intercept", "Coefficient"].iloc[0]
    formula = f"{TARGET} = {money(intercept)} " + " ".join(coefficient_terms)

    top_under_cards = outlier_cards(top_under, "U")
    top_over_cards = outlier_cards(top_over, "O")
    actual_vs_predicted_visual = render_actual_vs_predicted(predictions)
    lag_decay_visual = render_lag_decay_chart(lag_decay)
    lag_heatmap_visual = render_lag_heatmap(lag_decay)
    time_lag_visual = render_time_lag_detail(yearly_lag, overall_lag, lag_summary)
    nonlinear_note = nonlinear_correlation_note(nonlinear_checks)
    pairwise_near_zero_note = (
        f"No numeric variable pair has near-zero Pearson correlation at abs(r) <= {NEAR_ZERO_CORRELATION_LIMIT:.2f}."
        if near_zero_pairwise_checks.empty
        else f"{len(near_zero_pairwise_checks):,} numeric variable pair(s) have near-zero Pearson correlation."
    )
    skew_test_data = add_error_percentages(predictions)
    feature_skewness_visual = skewness_distribution_cards(
        skew_test_data,
        skewness,
        "Test Feature Skewness",
        FEATURES,
    )
    error_skewness_visual = skewness_distribution_cards(
        skew_test_data,
        skewness,
        "Test Target and Error Skewness",
        ["Actual Revenue Impact", "Predicted Revenue Impact", "Error", "Error Percentage"],
    )
    decision_tree_visual = render_decision_tree(
        decision_tree["model"],
        decision_tree["tree_features"],
        decision_tree["transform_lookup"],
    )
    adoption_tree_visual = render_decision_tree(
        adoption_tree_model["model"],
        adoption_tree_model["tree_features"],
        adoption_tree_model["transform_lookup"],
        "Supervised AI Adoption Decision Tree",
        "This tree is trained with ai_adoption_level as the supervised target. Each split predicts the adoption level from business and AI maturity variables.",
        number,
    )
    feature_range_visual = render_feature_range_overview(
        decision_tree["feature_map"],
        decision_tree["rules"],
    )
    separated_feature_tree_visual = render_separated_feature_tree_visual(
        separated_feature_splits,
        decision_tree["feature_map"],
    )
    separated_feature_visual = render_separated_feature_splits(separated_feature_splits)
    xgboost_ensemble_visual = render_xgboost_ensemble_visual(
        xgboost_model["importance"],
        xgboost_model["rules"],
        xgboost_model["split_audit"],
        tree_model_selection,
    )
    decision_tree_note = (
        "Model selection compares linear regression, a single decision tree, and XGBoost. "
        f"Final selected model: {html.escape(final_model_name)} by held-out test R Square."
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Regression Analysis</title>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; background: #f4f6f8; color: #17202a; }}
        header {{ padding: 24px 32px 16px; background: #ffffff; border-bottom: 1px solid #d8dee4; }}
        h1 {{ margin: 0 0 10px; font-size: 30px; }}
        h2 {{ margin: 0 0 12px; font-size: 20px; }}
        h3 {{ margin: 0 0 8px; font-size: 16px; }}
        h4 {{ margin: 0 0 10px; font-size: 14px; }}
        p {{ margin: 0 0 12px; color: #5b6670; line-height: 1.45; }}
        a {{ color: #1769aa; text-decoration: none; }}
        .page {{ padding: 20px 32px 32px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }}
        .summary-card, .panel {{ background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; padding: 16px; }}
        .summary-card span {{ display: block; color: #5b6670; font-size: 13px; margin-bottom: 6px; }}
        .summary-card strong {{ font-size: 24px; }}
        .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }}
        .toolbar button, .toolbar a {{ border: 1px solid #b7c2cc; background: #ffffff; color: #17202a; border-radius: 6px; padding: 9px 12px; cursor: pointer; font-size: 14px; }}
        .toolbar button.active {{ background: #17202a; color: #ffffff; border-color: #17202a; }}
        .formula {{ overflow-x: auto; background: #eef3f7; border: 1px solid #d8dee4; border-radius: 6px; padding: 12px; white-space: nowrap; }}
        .sheet {{ display: none; margin-top: 14px; }}
        .sheet.active {{ display: block; }}
        .excel-table {{ width: 100%; border-collapse: collapse; background: #ffffff; font-size: 13px; white-space: nowrap; }}
        .excel-table th, .excel-table td {{ border: 1px solid #d8dee4; padding: 8px 10px; text-align: right; }}
        .excel-table th {{ background: #eef3f7; }}
        .excel-table th:first-child, .excel-table td:first-child {{ text-align: left; }}
        .prediction-image {{ width: 100%; max-width: 1120px; display: block; background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; }}
        .calculated-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; margin-bottom: 14px; }}
        .calculated-chart {{ background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; margin: 0 0 14px; }}
        .calculated-svg {{ width: 100%; height: auto; display: block; }}
        .tree-scroll {{ overflow-x: auto; }}
        .tree-svg {{ min-width: 1280px; }}
        .tree-link {{ stroke: #b7c2cc; stroke-width: 1.4; }}
        .tree-node {{ fill: #ffffff; stroke: #b7c2cc; stroke-width: 1.1; rx: 7; }}
        .tree-node.split {{ fill: #f7fbfb; stroke: #587b7f; }}
        .tree-node.leaf {{ fill: #fffaf4; stroke: #d08c60; }}
        .tree-node-title {{ fill: #17202a; font-size: 11px; font-weight: 700; }}
        .tree-node-text {{ fill: #44515c; font-size: 10px; }}
        .split-card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
        .split-card {{ background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; display: grid; gap: 10px; }}
        .split-card-head {{ display: grid; gap: 4px; }}
        .split-card-head span {{ color: #5b6670; font-size: 12px; }}
        .split-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
        .split-metrics span, .split-branches span {{ background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 6px; padding: 8px; font-size: 12px; line-height: 1.35; }}
        .split-metrics b, .split-branches b {{ display: block; color: #5b6670; font-size: 11px; margin-bottom: 3px; }}
        .split-branches {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
        .model-score-visual {{ display: grid; gap: 10px; margin: 12px 0 16px; }}
        .model-score-row {{ display: grid; grid-template-columns: 190px minmax(160px, 1fr) 70px; gap: 10px; align-items: center; }}
        .model-score-row strong {{ font-size: 13px; }}
        .model-score-row b {{ font-size: 13px; text-align: right; }}
        .model-score-track, .xgb-importance-track {{ display: block; height: 13px; background: #eef3f7; border: 1px solid #d8dee4; border-radius: 999px; overflow: hidden; }}
        .model-score-track i, .xgb-importance-track i {{ display: block; height: 100%; background: #587b7f; border-radius: 999px; }}
        .xgb-visual-grid {{ display: grid; grid-template-columns: minmax(340px, 1.05fr) minmax(340px, 0.95fr); gap: 14px; margin-top: 8px; }}
        .xgb-visual-panel {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; background: #ffffff; }}
        .xgb-importance-list {{ display: grid; gap: 9px; }}
        .xgb-importance-row {{ display: grid; grid-template-columns: 180px minmax(130px, 1fr) 58px 72px; gap: 9px; align-items: center; }}
        .xgb-importance-row strong, .xgb-importance-row b {{ font-size: 12px; }}
        .xgb-importance-row em {{ color: #5b6670; font-size: 11px; font-style: normal; text-align: right; }}
        .xgb-tree-grid {{ display: grid; grid-template-columns: repeat(20, 1fr); gap: 4px; margin: 4px 0 12px; }}
        .xgb-tree-tile {{ aspect-ratio: 1; min-width: 9px; border-radius: 2px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55); }}
        .xgb-feature-legend {{ display: flex; flex-wrap: wrap; gap: 8px 12px; color: #5b6670; font-size: 12px; }}
        .xgb-feature-legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
        .xgb-feature-legend i {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}
        .xgb-audit-strip {{ margin-top: 12px; }}
        .stacked-hero {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(320px, 0.9fr); gap: 14px; align-items: stretch; margin-bottom: 14px; }}
        .stacked-hero > div, .stacked-card {{ background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; }}
        .stacked-formula {{ overflow-x: auto; background: #eef3f7; border: 1px solid #d8dee4; border-radius: 6px; padding: 10px; font-size: 13px; white-space: nowrap; }}
        .stacked-card-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
        .stacked-card span {{ display: block; color: #5b6670; font-size: 12px; margin-bottom: 6px; }}
        .stacked-card strong {{ font-size: 22px; }}
        .stacked-visual-grid {{ display: grid; grid-template-columns: minmax(420px, 1.25fr) minmax(320px, 0.75fr); gap: 14px; }}
        .stacked-point {{ fill: #36d8ff; opacity: 0.20; }}
        .stacked-regression-line {{ stroke: #ff65c8; stroke-width: 3; }}
        .stacked-histogram {{ max-height: 310px; }}
        .residual-bar.under {{ fill: #d04a3a; opacity: 0.82; }}
        .residual-bar.over {{ fill: #2f6fbb; opacity: 0.82; }}
        .stacked-legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
        .stacked-legend span {{ display: inline-flex; align-items: center; gap: 6px; color: #5b6670; font-size: 12px; }}
        .stacked-legend i {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}
        .svg-plot-bg {{ fill: #fbfcfd; stroke: #d8dee4; stroke-width: 1; }}
        .svg-grid {{ stroke: #d8dee4; stroke-width: 1; opacity: 0.75; }}
        .svg-axis {{ stroke: #17202a; stroke-width: 1.2; }}
        .svg-diagonal {{ stroke: #17202a; stroke-width: 1.5; stroke-dasharray: 7 5; }}
        .svg-tick {{ fill: #5b6670; font-size: 12px; }}
        .svg-axis-label {{ fill: #17202a; font-size: 13px; font-weight: 700; }}
        .svg-marker-label {{ font-size: 12px; font-weight: 700; }}
        .svg-cell-label {{ fill: #17202a; font-size: 13px; font-weight: 700; }}
        .scatter-point {{ fill: #587b7f; opacity: 0.24; }}
        .svg-line {{ fill: none; stroke-width: 2.4; }}
        .current-line {{ stroke: #587b7f; }}
        .prior-line {{ stroke: #d08c60; }}
        .current-fill {{ fill: #587b7f; font-weight: 700; }}
        .prior-fill {{ fill: #d08c60; font-weight: 700; }}
        .chart-caption {{ color: #5b6670; font-size: 12px; margin-top: 8px; }}
        .graph-note, .note {{ margin: 0 0 12px; color: #5b6670; }}
        .legend, .outlier-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
        .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
        .legend i {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}
        .low-error-key {{ background: #440154; }}
        .mid-error-key {{ background: #21918c; }}
        .high-error-key {{ background: #fde725; }}
        .scatter-key {{ background: #587b7f; }}
        .under-error-key {{ background: #d04a3a; }}
        .over-error-key {{ background: #2f6fbb; border-radius: 2px !important; }}
        .outlier-legend {{ margin-top: 18px; }}
        .outlier-section {{ margin-top: 16px; }}
        .outlier-item {{ background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; min-width: 210px; display: grid; gap: 4px; }}
        .outlier-item.under {{ border-left: 4px solid #d04a3a; }}
        .outlier-item.over {{ border-left: 4px solid #2f6fbb; }}
        .outlier-vars {{ display: grid; gap: 3px; margin-top: 8px; font-size: 12px; color: #44515c; }}
        .skew-distribution {{ margin: 0 0 16px; }}
        .skew-card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }}
        .skew-card {{ background: #ffffff; border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; }}
        .skew-card-head {{ display: grid; gap: 4px; margin-bottom: 8px; }}
        .skew-card-head span {{ color: #5b6670; font-size: 12px; }}
        .skew-svg {{ width: 100%; height: auto; display: block; background: #fbfcfd; border: 1px solid #eef3f7; border-radius: 6px; }}
        .skew-bars rect {{ fill: #7aa0a6; }}
        .axis-line {{ stroke: #8a98a6; stroke-width: 1; }}
        .median-marker {{ stroke: #2f6fbb; stroke-width: 2; }}
        .mean-marker {{ stroke: #d08c60; stroke-width: 2; stroke-dasharray: 4 3; }}
        .skew-axis, .skew-legend {{ display: flex; justify-content: space-between; gap: 8px; color: #5b6670; font-size: 11px; margin-top: 6px; }}
        .skew-legend {{ justify-content: flex-start; flex-wrap: wrap; }}
        .skew-legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
        .skew-legend i {{ width: 12px; height: 3px; display: inline-block; }}
        .percentile-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(74px, 1fr)); gap: 6px; margin-top: 10px; }}
        .percentile-strip span {{ background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 6px; padding: 6px; display: grid; gap: 3px; color: #17202a; font-size: 11px; }}
        .percentile-strip b {{ color: #5b6670; font-size: 10px; }}
        .median-key {{ background: #2f6fbb; }}
        .mean-key {{ background: #d08c60; }}
        .table-wrap {{ overflow-x: auto; }}
        {REGRESSION_THEME_CSS}
        {agent_widget_styles()}
        @media (max-width: 860px) {{
            .xgb-visual-grid {{ grid-template-columns: 1fr; }}
            .stacked-hero, .stacked-visual-grid {{ grid-template-columns: 1fr; }}
            .model-score-row, .xgb-importance-row {{ grid-template-columns: 1fr; }}
            .model-score-row b, .xgb-importance-row em {{ text-align: left; }}
        }}
        @media (max-width: 620px) {{ header, .page {{ padding-left: 14px; padding-right: 14px; }} }}
    </style>
</head>
<body>
    <header>
        <h1>Regression Analysis</h1>
        <p>Excel-style output for the revenue impact regression model.</p>
    </header>
    <div class="page">
        <section class="summary-grid">
            <div class="summary-card"><span>Selected Model</span><strong>{html.escape(final_model_name)}</strong></div>
            <div class="summary-card"><span>Final Test R Square</span><strong>{float(final_metric_lookup.loc["R Square", "Test Data"]):.4f}</strong></div>
            <div class="summary-card"><span>Final Test RMSE</span><strong>{money(final_metric_lookup.loc["RMSE", "Test Data"])}</strong></div>
            <div class="summary-card"><span>Test Observations</span><strong>{int(final_metric_lookup.loc["Observations", "Test Data"]):,}</strong></div>
        </section>
        <section class="panel">
            <h2>Linear Regression Formula</h2>
            <p>Baseline formula learned from the 70% training data after skew transformation and RobustScaler normalization.</p>
            <div class="formula">{formula}</div>
        </section>
        <div class="toolbar">
            <a href="dashboard.html">Back to Dashboard</a>
            <button class="active" data-sheet="stats">Training vs Test Stats</button>
            <button data-sheet="pipeline-audit">Pipeline Audit</button>
            <button data-sheet="skew-transform">Skew Transform</button>
            <button data-sheet="prediction-graph">Actual vs Predicted Graph</button>
            <button data-sheet="nonlinear-correlations">Nonlinear Correlations</button>
            <button data-sheet="time-lag">Time Lag Data</button>
            <button data-sheet="lag-model">Lagged Variable Model</button>
            <button data-sheet="error-cases">Largest Error Cases</button>
            <button data-sheet="error-percentages">Error Percentages</button>
            <button data-sheet="variable-skewness">Variable Skewness</button>
            <button data-sheet="decision-tree">Decision Tree</button>
            <button data-sheet="adoption-xgboost">Adoption Tree</button>
            <button data-sheet="stacked-xgboost">Stacked Ensemble</button>
            <button data-sheet="anova">ANOVA</button>
            <button data-sheet="coefficients">Coefficients</button>
            <button data-sheet="predictions">Prediction Table</button>
        </div>
        <p class="note">Target: revenue_impact. The graph and error cases use unseen test data.</p>
        <section class="sheet active table-wrap" id="stats">{table_html(format_table(stats))}</section>
        <section class="sheet prediction-panel" id="skew-transform">
            <div class="graph-note">Skew transforms are fitted only on the 70% training rows, then applied to both training and test data. The dependent variable remains revenue_impact.</div>
            <div class="table-wrap">{table_html(format_table(skew_transform_audit))}</div>
        </section>
        <section class="sheet prediction-panel" id="prediction-graph">
            <div class="graph-note">Calculated from the held-out test set. U1-U5 mark the largest underpredictions; O1-O5 mark the largest overpredictions.</div>
            {actual_vs_predicted_visual}
            <div class="legend">
                <span><i class="scatter-key"></i>Test rows</span>
                <span><i class="under-error-key"></i>Top underpredicted</span>
                <span><i class="over-error-key"></i>Top overpredicted</span>
            </div>
            <div class="outlier-legend">
                <div class="outlier-section">
                    <h3>Top 5 Underpredicted Outliers</h3>
                    <p>Actual revenue impact is higher than the model prediction.</p>
                    <div class="outlier-grid">{top_under_cards}</div>
                </div>
                <div class="outlier-section">
                    <h3>Top 5 Overpredicted Outliers</h3>
                    <p>Predicted revenue impact is higher than actual revenue impact.</p>
                    <div class="outlier-grid">{top_over_cards}</div>
                </div>
            </div>
        </section>
        <section class="sheet prediction-panel" id="nonlinear-correlations">
            <div class="graph-note">{nonlinear_note}</div>
            <h3>Nonlinear Correlation Checks</h3>
            <div class="table-wrap">{table_html(format_table(nonlinear_checks))}</div>
            <h3>Near-Zero Pairwise Checks</h3>
            <div class="graph-note">{pairwise_near_zero_note}</div>
            <div class="table-wrap">{table_html(format_table(near_zero_pairwise_checks))}</div>
        </section>
        <section class="sheet prediction-panel" id="time-lag">
            <div class="graph-note">Lag correlations compare current revenue impact with each variable from the same year through five prior years.</div>
            {lag_decay_visual}
            {lag_heatmap_visual}
            <h3>Correlation Decay Data</h3>
            <div class="table-wrap">{table_html(format_table(lag_decay))}</div>
            <h3>One-Year Lag Detail</h3>
            {time_lag_visual}
            <h3>Lag Summary</h3>
            <div class="table-wrap">{table_html(format_table(lag_summary))}</div>
            <h3>Overall Lag Correlations</h3>
            <div class="table-wrap">{table_html(format_table(overall_lag))}</div>
            <h3>Yearly Lag Correlations</h3>
            <div class="table-wrap">{table_html(format_table(yearly_lag))}</div>
        </section>
        <section class="sheet prediction-panel" id="lag-model">
            <div class="graph-note">The current + lag model predicts current revenue impact from current variables plus one-year-lagged variables. The split is chronological: earlier years train the model, later years test it.</div>
            <h3>Current + Lag Pipeline Audit</h3>
            <div class="table-wrap">{table_html(format_table(current_plus_lag_model["audit"]))}</div>
            <h3>Current + Lag Stats</h3>
            <div class="table-wrap">{table_html(format_table(current_plus_lag_model["stats"]))}</div>
            <h3>Lag Component by Variable</h3>
            <div class="table-wrap">{table_html(format_table(current_plus_lag_model["components"]))}</div>
            <h3>All Current + Lag Coefficients</h3>
            <div class="table-wrap">{table_html(format_table(current_plus_lag_model["coefficients"]))}</div>
            <h3>Lag-Only Model Check</h3>
            <div class="table-wrap">{table_html(format_table(lag_model["stats"]))}</div>
        </section>
        <section class="sheet table-wrap" id="error-cases">{table_html(format_table(largest))}</section>
        <section class="sheet prediction-panel" id="error-percentages">
            <div class="graph-note">Directional outliers with feature values and signed error percentage. Positive means underpredicted; negative means overpredicted.</div>
            <div class="table-wrap">{table_html(format_table(error_percentages))}</div>
        </section>
        <section class="sheet prediction-panel" id="variable-skewness">
            <div class="graph-note">Skewness is computed from held-out test data only. Positive means a longer right tail; negative means a longer left tail.</div>
            <h3>Feature Percentile Ranges</h3>
            <div class="table-wrap">{table_html(format_table(feature_percentiles))}</div>
            {feature_skewness_visual}
            {error_skewness_visual}
            <div class="table-wrap">{table_html(format_table(skewness))}</div>
        </section>
        <section class="sheet prediction-panel" id="decision-tree">
            <div class="graph-note">{decision_tree_note}</div>
            {xgboost_ensemble_visual}
            {feature_range_visual}
            {decision_tree_visual}
            {separated_feature_tree_visual}
            {separated_feature_visual}
            <h3>Model Selection Check</h3>
            <div class="table-wrap">{table_html(format_table(tree_model_selection))}</div>
            <h3>Final Selected Regression Stats</h3>
            <div class="table-wrap">{table_html(format_table(final_selected_stats))}</div>
            <h3>XGBoost Tree Ensemble Stats</h3>
            <div class="table-wrap">{table_html(format_table(xgboost_model["stats"]))}</div>
            <h3>XGBoost Skew-Aware Feature Inputs</h3>
            <div class="table-wrap">{table_html(format_table(xgboost_model["feature_map"]))}</div>
            <h3>XGBoost Split Stop Audit</h3>
            <div class="table-wrap">{table_html(format_table(xgboost_model["split_audit"]))}</div>
            <h3>XGBoost Feature Importance</h3>
            <div class="table-wrap">{table_html(format_table(xgboost_model["importance"]))}</div>
            <h3>XGBoost Tree Rules</h3>
            <div class="table-wrap">{table_html(format_table(xgboost_model["rules"].head(300)))}</div>
            <h3>Decision Tree Stats</h3>
            <div class="table-wrap">{table_html(format_table(decision_tree["stats"]))}</div>
            <h3>Skew-Aware Feature Inputs</h3>
            <div class="table-wrap">{table_html(format_table(decision_tree["feature_map"]))}</div>
            <h3>Feature Importance</h3>
            <div class="table-wrap">{table_html(format_table(decision_tree["importance"]))}</div>
            <h3>Decision Tree Rules</h3>
            <div class="table-wrap">{table_html(format_table(decision_tree["rules"]))}</div>
            <h3>Separated Variable Split Table</h3>
            <div class="table-wrap">{table_html(format_table(separated_feature_splits))}</div>
        </section>
        <section class="sheet prediction-panel" id="stacked-xgboost">
            <div class="graph-note">This stacked ensemble combines three base learners: baseline linear regression, decision tree, and XGBoost. These are also called base models, and sometimes weak learners in ensemble literature. The meta-regression is trained on out-of-fold base learner predictions, then tested on the held-out 30% split.</div>
            {stacked_xgboost["visual"]}
            <h3>Stacked Ensemble Regression Audit</h3>
            <div class="table-wrap">{table_html(format_table(stacked_xgboost["audit"]))}</div>
            <h3>Stacked Ensemble Regression Stats</h3>
            <div class="table-wrap">{table_html(format_table(stacked_xgboost["stats"]))}</div>
            <h3>Stacked Ensemble Coefficients</h3>
            <div class="table-wrap">{table_html(format_table(stacked_xgboost["coefficients"]))}</div>
            <h3>Stacked Ensemble Largest Error Cases</h3>
            <div class="table-wrap">{table_html(format_table(stacked_xgboost["largest"]))}</div>
            <h3>Stacked Ensemble Prediction Sample</h3>
            <div class="table-wrap">{table_html(format_table(stacked_xgboost["predictions"].head(100)))}</div>
        </section>
        <section class="sheet prediction-panel" id="adoption-xgboost">
            <div class="graph-note">This section uses supervised models that target ai_adoption_level. The visible tree is a readable decision tree; the XGBoost tables remain as the stronger boosted benchmark.</div>
            {adoption_tree_visual}
            <h3>Supervised Adoption Tree Stats</h3>
            <div class="table-wrap">{table_html(format_table(adoption_tree_model["stats"]))}</div>
            <h3>Supervised Adoption Tree Feature Inputs</h3>
            <div class="table-wrap">{table_html(format_table(adoption_tree_model["feature_map"]))}</div>
            <h3>Supervised Adoption Tree Feature Importance</h3>
            <div class="table-wrap">{table_html(format_table(adoption_tree_model["importance"]))}</div>
            <h3>Supervised Adoption Tree Rules</h3>
            <div class="table-wrap">{table_html(format_table(adoption_tree_model["rules"]))}</div>
            <h3>Supervised Adoption Tree Largest Error Cases</h3>
            <div class="table-wrap">{table_html(format_table(adoption_tree_model["largest"]))}</div>
            <h3>Adoption XGBoost Stats</h3>
            <div class="table-wrap">{table_html(format_table(adoption_xgboost_model["stats"]))}</div>
            <h3>Adoption XGBoost Split Audit</h3>
            <div class="table-wrap">{table_html(format_table(adoption_xgboost_model["split_audit"]))}</div>
            <h3>Adoption XGBoost Feature Inputs</h3>
            <div class="table-wrap">{table_html(format_table(adoption_xgboost_model["feature_map"]))}</div>
            <h3>Adoption XGBoost Feature Importance</h3>
            <div class="table-wrap">{table_html(format_table(adoption_xgboost_model["importance"]))}</div>
            <h3>Adoption XGBoost Largest Error Cases</h3>
            <div class="table-wrap">{table_html(format_table(adoption_xgboost_model["largest"]))}</div>
            <h3>Adoption XGBoost Tree Rules</h3>
            <div class="table-wrap">{table_html(format_table(adoption_xgboost_model["rules"].head(300)))}</div>
            <h3>Adoption XGBoost Prediction Sample</h3>
            <div class="table-wrap">{table_html(format_table(adoption_xgboost_model["predictions"].head(100)))}</div>
        </section>
        <section class="sheet table-wrap" id="anova">{table_html(format_table(anova))}</section>
        <section class="sheet table-wrap" id="pipeline-audit">{table_html(format_table(model_audit))}</section>
        <section class="sheet table-wrap" id="coefficients">{table_html(format_table(coefficients))}</section>
        <section class="sheet table-wrap" id="predictions">{table_html(format_table(export_predictions.head(100)))}</section>
    </div>
    {agent_widget_markup()}
    <script>
        {agent_widget_script()}
        document.querySelectorAll("[data-sheet]").forEach((button) => {{
            button.addEventListener("click", () => {{
                document.querySelectorAll("[data-sheet]").forEach((item) => item.classList.remove("active"));
                document.querySelectorAll(".sheet").forEach((sheet) => sheet.classList.remove("active"));
                button.classList.add("active");
                document.getElementById(button.dataset.sheet).classList.add("active");
            }});
        }});
        setupAgentWidget();
    </script>
</body>
</html>
"""
    OUTPUT_PATH.write_text(page, encoding="utf-8")
    return OUTPUT_PATH


if __name__ == "__main__":
    print(write_page())
