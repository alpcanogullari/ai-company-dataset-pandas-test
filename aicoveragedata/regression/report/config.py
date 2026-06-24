from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "aicoveragedata"
SITE_DIR = PACKAGE_DIR / "site"
DASHBOARD_DOWNLOAD_DIR = SITE_DIR / "downloads" / "dashboard"
DOWNLOAD_DIR = SITE_DIR / "downloads" / "regression"
DATA_PATH = DASHBOARD_DOWNLOAD_DIR / "full_dataset.csv"
OUTPUT_PATH = SITE_DIR / "regression_analysis.html"

TARGET = "revenue_impact"
FEATURES = [
    "ai_investment_usd",
    "automation_rate",
    "cost_savings",
    "employee_ai_training_hours",
    "deployment_count",
]
ADOPTION_TARGET = "ai_adoption_level"
ADOPTION_FEATURES = [
    "ai_investment_usd",
    "automation_rate",
    "cost_savings",
    "revenue_impact",
    "productivity_gain",
    "employee_ai_training_hours",
    "ai_maturity_score",
    "deployment_count",
    "total_benefit",
    "net_value",
    "roi",
]
LAG_DECAY_FEATURES = [
    "ai_investment_usd",
    "automation_rate",
    "cost_savings",
    "deployment_count",
    "employee_ai_training_hours",
]
TREE_MAX_DEPTH = 5
TREE_MIN_SAMPLES_LEAF = 500
TREE_MIN_NODE_GAIN = 0.05
XGBOOST_TREES = 200
XGBOOST_MAX_DEPTH = 5
XGBOOST_LEARNING_RATE = 0.04
NEAR_ZERO_CORRELATION_LIMIT = 0.05
CORRELATION_EXCLUDE_COLUMNS = ["company_id", "year"]
