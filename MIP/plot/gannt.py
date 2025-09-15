import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from schedule import schedule

plt.rcParams["font.family"] = "Times New Roman"

jobs = ["J1","J2","J3","J4","J5"]
palette_normal = {
    "J1": "#66AAD1",
    "J2": "#F0C566",
    "J3": "#66C5AB",
    "J4": "#E0AFCA",
    "J5": "#E69E66",
}
palette_high = {
    "J1": "#00446B",
    "J2": "#8A5F00",
    "J3": "#005F45",
    "J4": "#7A4964",
    "J5": "#803800",
}

machines = ["M3","M2","M1"]
height = 0.5
machine_y = {m: i for i, m in enumerate(reversed(machines))}


fig, ax = plt.subplots(figsize=(12, 3))

for m, j, s, e, hp in schedule:
    y = machine_y[m]
    width = e - s
    color = palette_high[j] if hp else palette_normal[j]
    rect = Rectangle((s, y - height/2), width, height, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ccolor = 'white' if hp else 'black'
    ax.text(s + width/2, y, j, ha='center', va='center', fontsize=8, weight='bold',
            color=ccolor)

boundaries = [0, 15, 45, 65]
for b in boundaries:
    ax.axvline(b, color='red', linestyle='--', linewidth=1)

ax.set_yticks([machine_y[m] for m in machines])
ax.set_yticklabels(machines, fontsize=10, fontweight='bold')

ax.set_xlim(0, 66)
ax.set_xticks(range(0, 66, 15))
ax.set_ylim(-1, len(machines))
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()