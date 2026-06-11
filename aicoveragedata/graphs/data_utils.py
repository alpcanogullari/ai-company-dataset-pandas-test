from pathlib import Path

import kagglehub
import pandas as pd


DATASET = "hassangasem/corporate-ai-adoption-and-roi-dataset-20152035"


def load_dataset():
    path = kagglehub.dataset_download(DATASET)
    csv_path = next(Path(path).glob("*.csv"))
    return pd.read_csv(csv_path)
