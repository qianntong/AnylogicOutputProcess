import pandas as pd
import matplotlib.pyplot as plt

file_path = "/Users/qianqiantong/PycharmProjects/AnylogicOutputProcess/output/same_throughput_fitting_results.xlsx"
df = pd.read_excel(file_path)

sensitivity = "Rows (M)"
group_columns = [col for col in ["Rows (M)", "Cols (N)", "Throughput (k)", "Vehicle Type", "Block Length (nr)"] if col != sensitivity]

df_filtered = df[df.groupby(group_columns)[sensitivity].transform(lambda x: x.nunique() == 2)]

color_map = {1: "blue", 2: "red"}
df_filtered["Color"] = df_filtered[sensitivity].map(color_map)

metrics = ["Power a", "Power b", "Exp a", "Exp b"]
vehicle_types = df_filtered["Vehicle Type"].unique()

for vehicle in vehicle_types:
    df_vehicle = df_filtered[df_filtered["Vehicle Type"] == vehicle]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for _, row in df_vehicle.iterrows():
            ax.scatter(row["Hostler"], row[metric], color=row["Color"], alpha=0.7, edgecolors="k")

        ax.set_xlabel("Hostler")
        ax.set_ylabel(metric)
        ax.set_title(f"{vehicle}: Effect of {sensitivity} on {metric}")
        ax.grid(True)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Block Length = 1"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Block Length = 2")]
    fig.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.show()
