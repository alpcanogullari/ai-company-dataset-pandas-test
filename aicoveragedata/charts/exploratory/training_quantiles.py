from aicoveragedata.data import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
df = load_dataset()

df["training_group"] = pd.qcut(
    df["employee_ai_training_hours"],
    q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"]
)


train_quantiles = df.groupby("training_group", observed=True)["productivity_gain"].agg(
    mean="mean",
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75)
)

train_quantiles["lower_error"] = train_quantiles["mean"] - train_quantiles["q1"]
train_quantiles["upper_error"] = train_quantiles["q3"] - train_quantiles["mean"] 

plt.figure(figsize=(10, 6))
plt.bar(
    train_quantiles.index,
    train_quantiles["mean"],
    yerr=[train_quantiles["lower_error"], train_quantiles["upper_error"]],
    capsize=3
)

plt.title("Productivity Gain by AI Training Hours Quantile")
plt.xlabel("AI Training Hours Group")
plt.ylabel("Productivity Gain")
plt.grid(axis="y")
plt.show()

df["roi"] = (df["cost_savings"] + df["revenue_impact"]) / df["ai_investment_usd"]

plt.figure(figsize=(10, 6))
df.boxplot(column="roi", by="training_group", showfliers=False)
plt.title("ROI Distribution by AI Training Hours Quantile")
plt.suptitle("")
plt.xlabel("AI Training Hours Group")
plt.ylabel("ROI: Value Per Dollar Invested")
plt.grid(axis="y")
plt.show()

plt.figure(figsize=(10, 6))
df.boxplot(column="productivity_gain", by="training_group", showfliers=False)
plt.title("Productivity Gain Distribution by AI Training Hours Quantile")
plt.suptitle("")
plt.xlabel("AI Training Hours Group")
plt.ylabel("Productivity Gain")
plt.grid(axis="y")
plt.show()
