import sys
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Allow this file to run directly from the regression folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from aicoveragedata.regression.legacy_utils import (
    load_regression_data,
    print_dataset_summary,
)


OUTPUT_DIR = PROJECT_ROOT / "aicoveragedata" / "site" / "downloads" / "regression"
TARGET_CORRELATION_IMAGE = OUTPUT_DIR / "target_correlation.png"
FEATURE_CORRELATION_IMAGE = OUTPUT_DIR / "feature_correlation_heatmap.png"


def interpret_strength(value):
    absolute_value = abs(value)

    if absolute_value >= 0.70:
        return "strong"
    if absolute_value >= 0.40:
        return "moderate"
    if absolute_value >= 0.20:
        return "weak"
    return "very weak"


def interpret_direction(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "no direction"


def build_target_correlation_table(X, y, target_column=None):
    # Compare each independent variable with the dependent variable.
    model_df = X.copy()
    target_name = target_column or y.name or "target"
    model_df[target_name] = y

    rows = []
    for feature in X.columns:
        pearson = model_df[feature].corr(model_df[target_name], method="pearson")
        spearman = model_df[feature].corr(model_df[target_name], method="spearman")

        rows.append(
            {
                "variable": feature,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "direction": interpret_direction(pearson),
                "strength": interpret_strength(pearson),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "pearson_correlation",
        key=abs,
        ascending=False,
    )


def build_feature_correlation_table(X):
    # Shows whether independent variables are highly related to each other.
    return X.corr(method="pearson")


def save_target_correlation_chart(correlation_table, output_path=TARGET_CORRELATION_IMAGE, target_column="target"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_data = correlation_table.sort_values("pearson_correlation")
    colors = [
        "#2f7d5b" if value >= 0 else "#b54d4d"
        for value in chart_data["pearson_correlation"]
    ]

    plt.figure(figsize=(9, 5))
    plt.barh(chart_data["variable"], chart_data["pearson_correlation"], color=colors)
    plt.axvline(0, color="#17202a", linewidth=1)
    plt.title(f"Correlation with {target_column}")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_feature_correlation_heatmap(feature_correlation, output_path=FEATURE_CORRELATION_IMAGE):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    image = plt.imshow(feature_correlation, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(image, label="Pearson correlation")
    plt.xticks(range(len(feature_correlation.columns)), feature_correlation.columns, rotation=45, ha="right")
    plt.yticks(range(len(feature_correlation.index)), feature_correlation.index)

    for row_index, row_name in enumerate(feature_correlation.index):
        for column_index, column_name in enumerate(feature_correlation.columns):
            value = feature_correlation.loc[row_name, column_name]
            plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Feature-to-feature correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    X, y, target_column, feature_columns = load_regression_data()
    target_correlation = build_target_correlation_table(X, y, target_column)
    feature_correlation = build_feature_correlation_table(X)

    save_target_correlation_chart(target_correlation, target_column=target_column)
    save_feature_correlation_heatmap(feature_correlation)

    print_dataset_summary(target_column, feature_columns)
    print()
    print("Correlation with dependent variable")
    print(f"Dependent variable: {target_column}")
    print(target_correlation.round(4).to_string(index=False))

    print()
    print("Feature-to-feature correlation")
    print(feature_correlation.round(4).to_string())
    print()
    print("Saved charts:")
    print(TARGET_CORRELATION_IMAGE)
    print(FEATURE_CORRELATION_IMAGE)


if __name__ == "__main__":
    main()
