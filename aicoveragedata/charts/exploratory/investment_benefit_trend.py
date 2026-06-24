from aicoveragedata.data import load_dataset
import matplotlib.pyplot as plt
df = load_dataset()


df["total_benefit"] = df["cost_savings"] + df["revenue_impact"]
df["benefit_ratio"] = df["total_benefit"] / df["ai_investment_usd"]
df["net_value"] = df["total_benefit"] - df["ai_investment_usd"]

value_trend = df.groupby("year").agg(
    avg_investment=("ai_investment_usd", "mean"),
    avg_total_benefit=("total_benefit", "mean"),
    avg_benefit_ratio=("benefit_ratio", "mean"),
    avg_net_value=("net_value", "mean")
)

plot_values = value_trend[["avg_investment", "avg_total_benefit"]] / 1_000_000

plt.figure(figsize=(12, 6))
plot_values.plot(kind="line", marker="o", ax=plt.gca())
plt.title("Average AI Investment vs Total Benefit by Year")
plt.xlabel("Year")
plt.ylabel("USD Millions")
plt.xticks(value_trend.index, value_trend.index.astype(int), rotation=45)
plt.grid(True)
plt.show()
