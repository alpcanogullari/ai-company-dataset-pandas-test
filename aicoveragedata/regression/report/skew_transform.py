import numpy as np
import pandas as pd

from .config import FEATURES
from .profiles import skew_direction, skew_strength

SKEW_TRANSFORM_THRESHOLD = 0.5


def build_skew_transform_plan(train_frame):
    rows = []
    for feature in FEATURES:
        series = pd.to_numeric(train_frame[feature], errors="coerce").dropna()
        skewness = series.skew() if len(series) > 2 else np.nan
        transform = "raw"
        parameter = np.nan

        if pd.notna(skewness) and skewness >= SKEW_TRANSFORM_THRESHOLD and series.min() >= 0:
            transform = "log1p"
        elif pd.notna(skewness) and skewness <= -SKEW_TRANSFORM_THRESHOLD:
            transform = "reflected_log1p"
            parameter = series.max()

        rows.append(
            {
                "Feature": feature,
                "Raw Training Skewness": skewness,
                "Raw Skew Direction": skew_direction(skewness),
                "Raw Skew Strength": skew_strength(skewness),
                "Transform": transform,
                "Transform Parameter": parameter,
            }
        )
    return pd.DataFrame(rows)


def apply_skew_transform(frame, plan):
    transformed = frame.copy()
    plan_lookup = plan.set_index("Feature")

    for feature, row in plan_lookup.iterrows():
        values = pd.to_numeric(transformed[feature], errors="coerce")
        transform = row["Transform"]

        if transform == "log1p":
            transformed[feature] = np.log1p(values.clip(lower=0))
        elif transform == "reflected_log1p":
            reflected = float(row["Transform Parameter"]) - values
            transformed[feature] = np.log1p(reflected.clip(lower=0))
        else:
            transformed[feature] = values

    return transformed


def build_skew_transform_audit(raw_train, transformed_train, raw_test, transformed_test, plan):
    rows = []
    for _, plan_row in plan.iterrows():
        feature = plan_row["Feature"]
        for split_name, raw_frame, transformed_frame in [
            ("Training Data", raw_train, transformed_train),
            ("Test Data", raw_test, transformed_test),
        ]:
            raw_series = pd.to_numeric(raw_frame[feature], errors="coerce").dropna()
            transformed_series = pd.to_numeric(transformed_frame[feature], errors="coerce").dropna()
            raw_skew = raw_series.skew() if len(raw_series) > 2 else np.nan
            transformed_skew = transformed_series.skew() if len(transformed_series) > 2 else np.nan
            rows.append(
                {
                    "Feature": feature,
                    "Split": split_name,
                    "Rows": len(transformed_series),
                    "Transform": plan_row["Transform"],
                    "Raw Skewness": raw_skew,
                    "Transformed Skewness": transformed_skew,
                    "Raw Skew Strength": skew_strength(raw_skew),
                    "Transformed Skew Strength": skew_strength(transformed_skew),
                    "Raw Mean": raw_series.mean(),
                    "Transformed Mean": transformed_series.mean(),
                    "Raw P50": raw_series.median(),
                    "Transformed P50": transformed_series.median(),
                }
            )
    return pd.DataFrame(rows)
