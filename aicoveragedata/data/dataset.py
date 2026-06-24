import pandas as pd
import kagglehub
from pathlib import Path

# just a simpler way to load dataset rather than putting the same data over and over again
DATASET = "hassangasem/corporate-ai-adoption-and-roi-dataset-20152035"
GENERATED_BUCKET_COLUMNS = ["training_group", "invest_group"]
LOCAL_DATASET = (
    Path(__file__).resolve().parents[1]
    / "site"
    / "downloads"
    / "dashboard"
    / "full_dataset.csv"
)

def load_dataset():
    if LOCAL_DATASET.exists():
        return pd.read_csv(LOCAL_DATASET).drop(columns=GENERATED_BUCKET_COLUMNS, errors="ignore")
    path = kagglehub.dataset_download(DATASET)
    csv_path = next(Path(path).glob("*.csv"))
    df = pd.read_csv(csv_path)
    return df
