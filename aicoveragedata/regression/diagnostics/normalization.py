import sys
from pathlib import Path

# Allow this file to run directly from the regression folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from aicoveragedata.regression.baseline import (
    build_pipeline,
    evaluate_model,
    print_coefficients,
    print_metrics,
    print_overfitting_check,
    split_data,
    train_model,
)
from aicoveragedata.regression.legacy_utils import (
    load_regression_data,
    print_dataset_summary,
)


def main():
    # Load features and target from our dataset.
    X, y, target_column, feature_columns = load_regression_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Build and train the normalization + regression pipeline.
    pipeline = build_pipeline()
    model = train_model(pipeline, X_train, y_train)

    # Calculate model performance on training and test data.
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    print_dataset_summary(target_column, feature_columns)
    print()
    print_metrics("Test performance", test_metrics)
    print_overfitting_check(train_metrics, test_metrics)
    print_coefficients(model, feature_columns)


if __name__ == "__main__":
    # Run the script only when this file is executed directly.
    main()
