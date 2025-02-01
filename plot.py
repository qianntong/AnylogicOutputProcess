import pandas as pd
import matplotlib.pyplot as plt


file_path = "data/all_data.xlsx"
df = pd.read_excel(file_path, sheet_name='sheet 1')

hostler_columns = [col for col in df.columns if "Hostler" in col]
truck_columns = [col for col in df.columns if "Truck" in col]
hostler_groups = sorted(set(col.split("_")[0] for col in hostler_columns))
truck_groups = sorted(set(col.split("_")[0] for col in truck_columns))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

# hostler plot
fig1, ax1 = plt.subplots(figsize=(8, 6))
for i, group in enumerate(hostler_groups):
    density_col = f"{group}_HostlerAvgDensity(veh/m)"
    speed_col = f"{group}_HostlerAvgSpeed(m/s)"
    if density_col in df.columns and speed_col in df.columns:
        ax1.scatter(df[density_col], df[speed_col], label=f"{group} hostlers", color=colors[i % len(colors)], alpha=0.7)

ax1.set_xlabel("Hostler Avg Density (veh/m)")
ax1.set_ylabel("Hostler Avg Speed (m/s)")
ax1.set_title("Hostler Speed vs Density")
ax1.legend()
ax1.grid(True)

# truck plot
fig2, ax2 = plt.subplots(figsize=(8, 6))
for i, group in enumerate(truck_groups):
    density_col = f"{group}_TruckAvgDensity(veh/m)"
    speed_col = f"{group}_TruckAvgSpeed(m/s)"
    if density_col in df.columns and speed_col in df.columns:
        ax2.scatter(df[density_col], df[speed_col], label=f"{group} trucks", color=colors[i % len(colors)], alpha=0.7)

ax2.set_xlabel("Truck Avg Density (veh/m)")
ax2.set_ylabel("Truck Avg Speed (m/s)")
ax2.set_title("Truck Speed vs Density")
ax2.legend()
ax2.grid(True)

plt.show()