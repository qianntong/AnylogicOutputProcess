import pandas as pd
import matplotlib.pyplot as plt

file_path = '/Users/qianqiantong/PycharmProjects/AnylogicOutputProcess/output/same_layout_fitting_results.xlsx'
df = pd.read_excel(file_path)

color_map = {30: "blue", 35: "green", 40: "red", 45: "yellow"}
metrics = ["Power a", "Power b", "Exp a", "Exp b"]
vehicle_types = df["Vehicle Type"].unique()

for vehicle in vehicle_types:
    df_vehicle = df[df["Vehicle Type"] == vehicle]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for throughput in df_vehicle["Throughput"].unique():
            df_subset = df_vehicle[df_vehicle["Throughput"] == throughput]
            ax.scatter(df_subset["Hostler"], df_subset[metric], color=color_map[throughput], alpha=0.7, edgecolors="k",
                       label=f"Throughput = {throughput}")

        ax.set_xlabel("Hostler")
        ax.set_ylabel(metric)
        ax.set_title(f"{vehicle}: Effect of Throughput on {metric}")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
