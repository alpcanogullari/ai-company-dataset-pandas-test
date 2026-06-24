from aicoveragedata.data import load_dataset
import matplotlib as plt
import pandas as pd
df = load_dataset()


df["roi"] = (df["cost_savings"] + df["revenue_impact"]) / df["ai_investment_usd"]
df["invest_group"] = pd.qcut(
    df["ai_investment_usd"],
    q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"]
)

# 
roi_quantiles = df.groupby("invest_group", observed=True)["roi"].agg(
    mean="mean",
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75)
)

roi_quantiles["lower_error"] = roi_quantiles["mean"] - roi_quantiles["q1"]
roi_quantiles["upper_error"] = roi_quantiles["q3"] - roi_quantiles["mean"]

plt.figure(figsize=(10, 6))
plt.bar(
    roi_quantiles.index,
    roi_quantiles["mean"],
    yerr=[roi_quantiles["lower_error"], roi_quantiles["upper_error"]],
    capsize=3
)

plt.title("ROI by AI Investment Quantile")
plt.xlabel("AI Investment Group")
plt.ylabel("ROI: Value Per Dollar Invested")
plt.grid(axis="y")
plt.show()

plt.figure(figsize=(10, 6))
df.boxplot(column="roi", by="invest_group", showfliers=False)
plt.title("ROI Distribution by AI Investment Quantile")
plt.suptitle("")
plt.xlabel("AI Investment Group")
plt.ylabel("ROI: Value Per Dollar Invested")
plt.grid(axis="y")
plt.show()
