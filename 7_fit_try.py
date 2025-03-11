import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def power_func(x, a, b):
    return a * x ** b

def exp_func(x, a, b):
    return a * np.exp(b * x)

def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def polynomial_func(x, a, b, c):
    return a * x ** 2 + b * x + c

def linear_func(x, a, b):
    return a * x + b

fit_functions = {
    "Power": power_func,
    "Exponential": exp_func,
    "Logistic": logistic_func,
    # "Polynomial": polynomial_func,
    # "Linear": linear_func
}

colors = {
    "Power": "green",
    "Exponential": "orange",
    "Logistic": "purple",
    # "Polynomial": "brown",
    # "Linear": "gray"
}

input_file = 'output/all_results.xlsx'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(input_file)

exclude_pattern = "2_2_19_150"

combo_patterns = set()
for col in df.columns:
    match = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)", col)
    if match:
        combo_pattern = match.group(0)
        if combo_pattern != exclude_pattern:
            combo_patterns.add(combo_pattern)

df_hostler_list = []
df_truck_list = []

for combo_pattern in combo_patterns:
    hostler_cols = [col for col in df.columns if re.search(f"^{combo_pattern}_\\d+_HostlerAvgDensity\\(veh/m\\)", col)]
    for density_col in hostler_cols:
        speed_col = density_col.replace("Density(veh/m)", "Speed(m/s)")
        if speed_col in df.columns:
            temp_df = df[[density_col, speed_col]].dropna()
            temp_df.columns = ['Density (veh/m)', 'Speed (m/s)']
            df_hostler_list.append(temp_df)

    truck_cols = [col for col in df.columns if re.search(f"^{combo_pattern}_\\d+_TruckAvgDensity\\(veh/m\\)", col) or re.search(f"^{combo_pattern}_TruckAvgDensity\\(veh/m\\)", col)]
    for density_col in truck_cols:
        speed_col = density_col.replace("Density(veh/m)", "Speed(m/s)")
        if speed_col in df.columns:
            temp_df = df[[density_col, speed_col]].dropna()
            temp_df.columns = ['Density (veh/m)', 'Speed (m/s)']
            df_truck_list.append(temp_df)

df_hostler = pd.concat(df_hostler_list, ignore_index=True) if df_hostler_list else pd.DataFrame()
df_truck = pd.concat(df_truck_list, ignore_index=True) if df_truck_list else pd.DataFrame()

if df_truck.empty and df_hostler.empty:
    print("Warning: No data found for the specified combinations")
else:
    plt.figure(figsize=(8, 6))

    # **Hostler**
    if not df_hostler.empty:
        x_hostler = df_hostler['Density (veh/m)'].values
        y_hostler = df_hostler['Speed (m/s)'].values
        sns.scatterplot(x=x_hostler, y=y_hostler, color='blue', label='Hostler Data')

        for name, func in fit_functions.items():
            try:
                if name == "Logistic":
                    L_initial = max(y_hostler)
                    x0_initial = np.median(x_hostler)
                    params, _ = curve_fit(func, x_hostler, y_hostler, p0=(L_initial, 1, x0_initial))
                    formula = f'{params[0]:.2f}/(1+e^(-{params[1]:.2f}(x-{params[2]:.2f})))'
                elif name == "Polynomial":
                    params, _ = curve_fit(func, x_hostler, y_hostler, p0=(1, 1, 1))
                    formula = f'{params[0]:.2f}x² + {params[1]:.2f}x + {params[2]:.2f}'
                else:
                    params, _ = curve_fit(func, x_hostler, y_hostler)
                    formula = f'{params[0]:.2f}x + {params[1]:.2f}' if name == "Linear" else f'{params[0]:.2f}x^{params[1]:.2f}' if name == "Power" else f'{params[0]:.2f}e^({params[1]:.2f}x)'

                y_fit = func(np.sort(x_hostler), *params)
                r2 = r2_score(y_hostler, func(x_hostler, *params))
                plt.plot(np.sort(x_hostler), y_fit, color=colors[name], label=f'Hostler {name}: {formula}, R²={r2:.2f}', linestyle='-')
            except Exception:
                pass

    # **Truck**
    if not df_truck.empty:
        x_truck = df_truck['Density (veh/m)'].values
        y_truck = df_truck['Speed (m/s)'].values
        sns.scatterplot(x=x_truck, y=y_truck, color='red', label='Truck Data')

        for name, func in fit_functions.items():
            try:
                if name == "Logistic":
                    L_initial = max(y_truck)
                    x0_initial = np.median(x_truck)
                    params, _ = curve_fit(func, x_truck, y_truck, p0=(L_initial, 1, x0_initial))
                    formula = f'{params[0]:.2f}/(1+e^(-{params[1]:.2f}(x-{params[2]:.2f})))'
                elif name == "Polynomial":
                    params, _ = curve_fit(func, x_truck, y_truck, p0=(1, 1, 1))
                    formula = f'{params[0]:.2f}x² + {params[1]:.2f}x + {params[2]:.2f}'
                else:
                    params, _ = curve_fit(func, x_truck, y_truck)
                    formula = f'{params[0]:.2f}x + {params[1]:.2f}' if name == "Linear" else f'{params[0]:.2f}x^{params[1]:.2f}' if name == "Power" else f'{params[0]:.2f}e^({params[1]:.2f}x)'

                y_fit = func(np.sort(x_truck), *params)
                r2 = r2_score(y_truck, func(x_truck, *params))
                plt.plot(np.sort(x_truck), y_fit, color=colors[name], label=f'Truck {name}: {formula}, R²={r2:.2f}', linestyle='--')
            except Exception:
                pass

    plt.title("Density-Speed Function\nAll Combinations (Excluding 2_2_19_150)")
    plt.xlabel('Density (veh/m)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, "All_Combinations_Combined.png")
    plt.savefig(output_path)
    plt.close()

print("\nDone!")
