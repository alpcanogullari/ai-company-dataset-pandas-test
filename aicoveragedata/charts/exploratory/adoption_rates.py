from aicoveragedata.data import load_dataset
df = load_dataset()
import matplotlib as plt

# sets up the matplotlib line graph that shows the ai_adoption_level over the years
yearly_adoption = df.groupby("year")["ai_adoption_level"].mean()
# checks the rate of change between two consecutive years regarding ai adoption
yearly_adoption_change = yearly_adoption.diff()
plt.figure(figsize=(12, 6))
ax = yearly_adoption.plot(kind= "line", marker= "o")
# labels for the axis
plt.title("Average AI Adoptions by Year")
plt.xlabel("Year")
plt.xticks(
    yearly_adoption.index,
    yearly_adoption.index.astype("int")
    )
plt.ylabel("AI Adoption Level")
plt.grid(True)

# prints the ai_adoption_level (y-coordinate) on the point
for year, value in yearly_adoption.items():
    ax.annotate(f"{value:.3f}", (year, value), textcoords="offset points", xytext=(0, 8), ha="center")

# prints the rate of change between the previous years, and alligns it with the line graph accordingly (dropna() removes null information)
for year, value in yearly_adoption_change.dropna().items():
    ax.annotate(f"+{value:.3f}", (year, yearly_adoption.loc[year]), textcoords="offset points", xytext=(0, -16), ha="center")

