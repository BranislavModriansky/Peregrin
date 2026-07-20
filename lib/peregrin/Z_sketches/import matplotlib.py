import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\modri\Desktop\Lab\Peregrin project\Data\Track stats 2026-02-22.csv")
print(df.columns)


x = df["Track length"].to_numpy(float)
y = df["Speed sem"].to_numpy(float)

fig, ax = plt.subplots()
pts = ax.scatter(x, y, s=6, alpha=0.5)
xy = np.column_stack([x, y])

def on_select(verts):
    path = Path(verts)
    inside = path.contains_points(xy)
    colors = np.where(inside, "red", "C0")
    pts.set_color(colors)
    fig.canvas.draw_idle()
    print(f"{inside.sum()} points inside gate")

selector = PolygonSelector(ax, on_select)  # click to add, double-click/esc to finish
plt.show()

