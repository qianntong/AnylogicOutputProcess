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

input_file = 'output/all_results.xlsx'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(input_file)

# exclude 150 combo
exclude_pattern = "2_2_19_150"

combo_patterns = set()
for col in df.columns:
    match = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)", col)
    if match:
        combo_pattern = match.group(0)  # Rows_Cols_BlockLen_Throughput
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

    if not df_hostler.empty:
        x_hostler = df_hostler['Density (veh/m)'].values
        y_hostler = df_hostler['Speed (m/s)'].values
        sns.scatterplot(x=x_hostler, y=y_hostler, color='blue', label='Hostler Data')

        try:
            power_params, _ = curve_fit(power_func, x_hostler, y_hostler, p0=(1, 0.5))
            power_y_fit = power_func(np.sort(x_hostler), *power_params)
            power_r2 = r2_score(y_hostler, power_func(x_hostler, *power_params))
            plt.plot(np.sort(x_hostler), power_y_fit, color='green', label=f'Hostler Power: y={power_params[0]:.2f}x^{power_params[1]:.2f}, R²={power_r2:.2f}', linestyle='-')
        except Exception:
            pass

        try:
            exp_params, _ = curve_fit(exp_func, x_hostler, y_hostler, p0=(1, 0.1))
            exp_y_fit = exp_func(np.sort(x_hostler), *exp_params)
            exp_r2 = r2_score(y_hostler, exp_func(x_hostler, *exp_params))
            plt.plot(np.sort(x_hostler), exp_y_fit, color='orange', label=f'Hostler Exp: y={exp_params[0]:.2f}e^({exp_params[1]:.2f}x), R²={exp_r2:.2f}', linestyle='-')
        except Exception:
            pass

    if not df_truck.empty:
        x_truck = df_truck['Density (veh/m)'].values
        y_truck = df_truck['Speed (m/s)'].values
        sns.scatterplot(x=x_truck, y=y_truck, color='red', label='Truck Data')

        try:
            power_params, _ = curve_fit(power_func, x_truck, y_truck, p0=(1, 0.5))
            power_y_fit = power_func(np.sort(x_truck), *power_params)
            power_r2 = r2_score(y_truck, power_func(x_truck, *power_params))
            plt.plot(np.sort(x_truck), power_y_fit, color='darkgreen', label=f'Truck Power: y={power_params[0]:.2f}x^{power_params[1]:.2f}, R²={power_r2:.2f}', linestyle='--')
        except Exception:
            pass

        try:
            exp_params, _ = curve_fit(exp_func, x_truck, y_truck, p0=(1, 0.1))
            exp_y_fit = exp_func(np.sort(x_truck), *exp_params)
            exp_r2 = r2_score(y_truck, exp_func(x_truck, *exp_params))
            plt.plot(np.sort(x_truck), exp_y_fit, color='darkorange', label=f'Truck Exp: y={exp_params[0]:.2f}e^({exp_params[1]:.2f}x), R²={exp_r2:.2f}', linestyle='--')
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
