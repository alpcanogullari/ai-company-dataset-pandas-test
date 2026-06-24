from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[2]
SITE_DIR = PACKAGE_DIR / "site"
DASHBOARD_DOWNLOAD_DIR = SITE_DIR / "downloads" / "dashboard"
REGRESSION_DOWNLOAD_DIR = SITE_DIR / "downloads" / "regression"
PROJECT_TEXT_SUFFIXES = {".py", ".html", ".md"}
PROJECT_EXCLUDE_PARTS = {"__pycache__", "downloads", "reports"}

FULL_DATASET = DASHBOARD_DOWNLOAD_DIR / "full_dataset.csv"

DASHBOARD_FILES = {
    "summary": DASHBOARD_DOWNLOAD_DIR / "summary.csv",
    "yearly_adoption": DASHBOARD_DOWNLOAD_DIR / "yearly_adoption.csv",
    "industry_adoption": DASHBOARD_DOWNLOAD_DIR / "industry_adoption.csv",
    "country_adoption": DASHBOARD_DOWNLOAD_DIR / "country_adoption.csv",
    "industry_profiles": DASHBOARD_DOWNLOAD_DIR / "industry_ai_profiles.csv",
    "country_profiles": DASHBOARD_DOWNLOAD_DIR / "country_ai_profiles.csv",
}

REGRESSION_FILES = {
    "skew_transformed_data": REGRESSION_DOWNLOAD_DIR / "regression_skew_transformed_model_data.csv",
    "skew_transform_audit": REGRESSION_DOWNLOAD_DIR / "regression_skew_transform_audit.csv",
    "statistics": REGRESSION_DOWNLOAD_DIR / "regression_statistics.csv",
    "coefficients": REGRESSION_DOWNLOAD_DIR / "regression_coefficients.csv",
    "target_correlations": REGRESSION_DOWNLOAD_DIR / "regression_target_correlations.csv",
    "feature_correlations": REGRESSION_DOWNLOAD_DIR / "regression_feature_correlations.csv",
    "model_selection": REGRESSION_DOWNLOAD_DIR / "regression_tree_model_selection.csv",
    "final_model": REGRESSION_DOWNLOAD_DIR / "regression_final_selected_model_statistics.csv",
    "xgboost_statistics": REGRESSION_DOWNLOAD_DIR / "regression_xgboost_statistics.csv",
    "xgboost_importance": REGRESSION_DOWNLOAD_DIR / "regression_xgboost_importance.csv",
    "xgboost_rules": REGRESSION_DOWNLOAD_DIR / "regression_xgboost_tree_rules.csv",
    "adoption_xgboost_statistics": REGRESSION_DOWNLOAD_DIR / "regression_adoption_xgboost_statistics.csv",
    "adoption_xgboost_importance": REGRESSION_DOWNLOAD_DIR / "regression_adoption_xgboost_importance.csv",
    "adoption_xgboost_rules": REGRESSION_DOWNLOAD_DIR / "regression_adoption_xgboost_tree_rules.csv",
    "adoption_tree_statistics": REGRESSION_DOWNLOAD_DIR / "regression_adoption_tree_statistics.csv",
    "adoption_tree_importance": REGRESSION_DOWNLOAD_DIR / "regression_adoption_tree_importance.csv",
    "adoption_tree_rules": REGRESSION_DOWNLOAD_DIR / "regression_adoption_tree_rules.csv",
    "stacked_ensemble_statistics": REGRESSION_DOWNLOAD_DIR / "regression_stacked_ensemble_statistics.csv",
    "stacked_ensemble_coefficients": REGRESSION_DOWNLOAD_DIR / "regression_stacked_ensemble_coefficients.csv",
    "stacked_ensemble_audit": REGRESSION_DOWNLOAD_DIR / "regression_stacked_ensemble_audit.csv",
    "stacked_xgboost_statistics": REGRESSION_DOWNLOAD_DIR / "regression_stacked_xgboost_statistics.csv",
    "stacked_xgboost_coefficients": REGRESSION_DOWNLOAD_DIR / "regression_stacked_xgboost_coefficients.csv",
    "stacked_xgboost_audit": REGRESSION_DOWNLOAD_DIR / "regression_stacked_xgboost_audit.csv",
    "decision_tree_statistics": REGRESSION_DOWNLOAD_DIR / "regression_decision_tree_statistics.csv",
    "decision_tree_importance": REGRESSION_DOWNLOAD_DIR / "regression_decision_tree_importance.csv",
    "largest_errors": REGRESSION_DOWNLOAD_DIR / "regression_largest_error_cases.csv",
}
