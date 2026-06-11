from data_utils  import load_dataset
import matplotlib as plt
df = load_dataset()

yearly_corr = df.groupby("year")[["ai_adoption_level", "productivity_gain"]].mean()
adoption_productivity_corr = yearly_corr["ai_adoption_level"].corr(yearly_corr["productivity_gain"])

yearly_corr.plot(
    kind="scatter",
    x="ai_adoption_level",
    y="productivity_gain",
    figsize=(10, 6)
)

plt.title(f"AI Adoption vs Productivity Gain | Correlation: {adoption_productivity_corr:.2f}")
plt.xlabel("Average AI Adoption Level")
plt.ylabel("Average Productivity Gain")
plt.grid(True)
plt.show()
