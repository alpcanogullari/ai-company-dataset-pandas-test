from data_utils import load_dataset
import matplotlib.pyplot as plt

df = load_dataset()

df["total_benefit"] = df["cost_savings"] + df["revenue_impact"]
df["net_value"] = df["total_benefit"] - df["ai_investment_usd"]

industry_net_value = df.groupby("industry")["net_value"].mean() / 1_000_000
industry_net_value = industry_net_value.sort_values()

plt.figure(figsize=(10, 6))
plt.barh(industry_net_value.index, industry_net_value)
plt.title("Average Net Value by Industry")
plt.xlabel("Average Net Value USD Millions")
plt.ylabel("Industry")
plt.grid(axis="x")
plt.show()
