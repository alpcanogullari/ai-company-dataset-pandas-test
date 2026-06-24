from aicoveragedata.data import load_dataset
import matplotlib.pyplot as plt
df = load_dataset()


df["total_benefit"] = df["revenue_impact"] + df["cost_savings"]

plt.figure(figsize=(10, 6))
plt.scatter(
    df["ai_investment_usd"] / 1_000_000,
    df["total_benefit"] / 1_000_000,
    alpha=0.6
)

plt.title("AI Investment vs Total Benefit")
plt.xlabel("AI Investment USD Millions")
plt.ylabel("Total Benefit USD Millions")
plt.grid(True)
plt.show()
