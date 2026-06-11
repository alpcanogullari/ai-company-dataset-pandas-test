
import pandas as pd
import kagglehub
from pathlib import Path

# just a simpler way to load dataset rather than putting the same data over and over again
DATASET = "hassangasem/corporate-ai-adoption-and-roi-dataset-20152035"

def load_dataset():
    path = kagglehub.dataset_download(DATASET)
    csv_path = next(Path(path).glob("*.csv"))
    df = pd.read_csv(csv_path)
    return df
