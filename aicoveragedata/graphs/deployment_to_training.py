from aicoveragedata.static.data_utils import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

df = load_dataset()

df["training_group"] = pd.qcut(
    df["employee_ai_training_hours"],
    q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"]
)

plt.figure(figsize=(10, 6))
avg_deployment = df.groupby("training_group", observed=True)["deployment_count"].mean()
plt.bar(avg_deployment.index, avg_deployment)
plt.title("Average Deployment Count by AI Training Hours Quartile")
plt.xlabel("AI Training Hours Quartile")
plt.ylabel("Average Deployment Count")
plt.grid(axis="y")
plt.show()
