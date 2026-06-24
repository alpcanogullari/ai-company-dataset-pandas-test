from aicoveragedata.data import load_dataset
import matplotlib.pyplot as plt

df = load_dataset()

plt.figure(figsize=(10, 6))
plt.scatter(
    df["automation_rate"],
    df["productivity_gain"],
    alpha=0.6
)
plt.title("Automation Rate vs Productivity Gain")
plt.xlabel("Automation Rate")
plt.ylabel("Productivity Gain")
plt.grid(True)
plt.show()
