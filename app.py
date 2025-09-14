import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
np.random.seed(42)
df = pd.DataFrame({
    "cell_id": [f"C{i:03}" for i in range(1, 21)],
    "temperature": np.random.randint(20, 55, 20),
    "voltage": np.round(np.random.uniform(3.0, 4.2, 20), 2),
    "current": np.round(np.random.uniform(0.5, 2.0, 20), 2),
})

# Detect defective cells
defective = df[df["temperature"] > 45]
percent = len(defective) / len(df) * 100

print(f"Summary: {len(defective)} defective out of {len(df)} ({percent:.1f}%)")
print(defective)

# Save outputs
defective.to_csv("defective_cells.csv", index=False)
plt.bar(defective["cell_id"], defective["temperature"])
plt.xticks(rotation=45)
plt.ylabel("Temp (°C)")
plt.title("Defective Cells")
plt.tight_layout()
plt.savefig("defective_cells_plot.png")

print("Saved defective_cells.csv and defective_cells_plot.png")

