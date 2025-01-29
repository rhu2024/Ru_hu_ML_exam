import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# Gebruik een variabele voor het bestandspad zonder hardcoded paden
csv_file = os.path.join(os.path.expanduser("~"), "Documents", "Ru_hu_ML_exam", "hypertuning_results", "results_total.csv")

df = pd.read_csv(csv_file)

# Selecteer de benodigde kolommen
x = df["config/hidden_size"]  # Hidden size (X-as)
y = df["config/num_blocks"]  # Number of layers (Y-as)
z = df["recall"]  # Recall als meetwaarde

# Maak een grid voor de contour plot
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)
Z = griddata((x, y), z, (X, Y), method='cubic')

# Plot de contouren
plt.figure(figsize=(12, 5))
plt.contourf(X, Y, Z, levels=20, cmap="plasma")
cbar = plt.colorbar()
cbar.set_label("Recall")

# Voeg contourlijnen en waarden toe
contours = plt.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
plt.clabel(contours, inline=True, fontsize=8)

# Voeg de originele meetpunten toe
plt.scatter(x, y, color='black')

# Labels en titel
plt.xlabel("Hidden Size")
plt.ylabel("Number of Layers")
plt.title("Contour Plot of Recall")
plt.show()



