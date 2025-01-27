import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Functie voor het maken van de contourplot
def plot_contour(df, x_col, y_col, z_col, start=0.7):
    # Filter de data
    x = df[x_col]
    y = df[y_col]
    z = df[z_col]

    # Maak een grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpoleer de data naar het grid
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Maak de plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xi, yi, zi, levels=np.linspace(start, z.max(), 20), cmap="viridis")
    plt.colorbar(contour, label=z_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Contour Plot van {z_col} gebaseerd op {x_col} en {y_col}')
    plt.show()

# Lees de CSV in
data = pd.read_csv("/Users/rubengoedings/Documents/Ru_hu_ML_exam/hypertuning_results/results_26jan.csv")

# Roep de functie aan
grid = data
plot_contour(grid, "config/hidden_size", "config/dropout", "accuracy", start=0.7)
