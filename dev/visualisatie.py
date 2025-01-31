import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# Bepaal het bestandspad dynamisch
base_dir = os.path.join(os.path.expanduser("~"), "Documents", "Ru_hu_ML_exam", "hypertuning_results")
csv_file = os.path.join(base_dir, "results_total.csv")
output_file = os.path.join(base_dir, "contour_plot.png")

# Lees de data in
df = pd.read_csv(csv_file)

# Selecteer de benodigde kolommen
x = df["config/hidden_size"]  # Hidden size (X-as)
y = df["config/num_blocks"]  # Number of layers (Y-as)
z = df["recall"]  # Recall als meetwaarde

# Maak een grid voor de contour plot
xi = np.linspace(x.min(), x.max(), 250)
yi = np.linspace(y.min(), y.max(), 250)
X, Y = np.meshgrid(xi, yi)
Z = griddata((x, y), z, (X, Y), method='cubic')

# Plot de contouren
plt.figure(figsize=(12, 5))
contour = plt.contourf(X, Y, Z, levels=50, cmap="plasma", alpha=0.9)
cbar = plt.colorbar()
cbar.set_label("Recall")

# Voeg contourlijnen en waarden toe
contours = plt.contour(X, Y, Z, levels=20, colors='black', linewidths=0.7)
plt.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

# Voeg de originele meetpunten toe
plt.scatter(x, y, color='black', edgecolors='black', s=20)

# Labels en titel
plt.xlabel("Hidden Size")
plt.ylabel("Number of Layers")
plt.title("Contour Plot of Recall with Enhanced Detail")

# Opslaan als PNG
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Contour plot opgeslagen als: {output_file}")



