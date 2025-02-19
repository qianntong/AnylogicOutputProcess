import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

combinations = [
    {'Rows': 2, 'Cols': 2, 'BlockLen': 19, 'Throughput': 150}
]

fit_results = []

for combo in combinations:
    combo_pattern = f"{combo['Rows']}_{combo['Cols']}_{combo['BlockLen']}_{combo['Throughput']}"
    truck_density_col = next((col for col in df.columns if col.startswith(f"{combo_pattern}_TruckAvgDensity")), None)
    truck_speed_col = next((col for col in df.columns if col.startswith(f"{combo_pattern}_TruckAvgSpeed")), None)

    df_truck = pd.DataFrame()
    if truck_density_col and truck_speed_col:
        df_truck = df[[truck_density_col, truck_speed_col]].dropna()
        df_truck.columns = ['Density', 'Speed']

    df_hostler = pd.DataFrame()
    hostler_df_list = []
    for hostler_num in [5, 10, 15]:
        hostler_density_col = next((col for col in df.columns if col.startswith(f"{combo_pattern}_{hostler_num}_HostlerAvgDensity")), None)
        hostler_speed_col = next((col for col in df.columns if col.startswith(f"{combo_pattern}_{hostler_num}_HostlerAvgSpeed")), None)
        if hostler_density_col and hostler_speed_col:
            temp_df = df[[hostler_density_col, hostler_speed_col]].dropna()
            temp_df.columns = ['Density', 'Speed']
            hostler_df_list.append(temp_df)
    if hostler_df_list:
        df_hostler = pd.concat(hostler_df_list, ignore_index=True)

    if df_truck.empty and df_hostler.empty:
        print(f"Warning: No data found for combination {combo}")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    if not df_hostler.empty:
        x = df_hostler['Density'].values
        y = df_hostler['Speed'].values

        sns.scatterplot(ax=axes[0], x=x, y=y, color='blue', label='Hostler Data')

        try:
            power_params, _ = curve_fit(power_func, x, y, p0=(1, 0.5))
            power_y_fit = power_func(x, *power_params)
            power_r2 = r2_score(y, power_y_fit)
            axes[0].plot(x, power_y_fit, color='green', label=f'Power Fit (R²={power_r2:.2f})')

            fit_results.append({
                'Combination': combo,
                'Type': 'Hostler',
                'Function': 'Power',
                'a': power_params[0],
                'b': power_params[1],
                'R²': power_r2
            })
        except Exception:
            pass

        try:
            exp_params, _ = curve_fit(exp_func, x, y, p0=(1, 0.1))
            exp_y_fit = exp_func(x, *exp_params)
            exp_r2 = r2_score(y, exp_y_fit)
            axes[0].plot(x, exp_y_fit, color='orange', label=f'Exp Fit (R²={exp_r2:.2f})')

            fit_results.append({
                'Combination': combo,
                'Type': 'Hostler',
                'Function': 'Exp',
                'a': exp_params[0],
                'b': exp_params[1],
                'R²': exp_r2
            })
        except Exception:
            pass

        axes[0].set_title('Hostler')
        axes[0].set_xlabel('Density (veh/m)')
        axes[0].set_ylabel('Speed (m/s)')
        axes[0].legend()

    if not df_truck.empty:
        x = df_truck['Density'].values
        y = df_truck['Speed'].values

        sns.scatterplot(ax=axes[1], x=x, y=y, color='blue', label='Truck Data')

        try:
            power_params, _ = curve_fit(power_func, x, y, p0=(1, 0.5))
            power_y_fit = power_func(x, *power_params)
            power_r2 = r2_score(y, power_y_fit)
            axes[1].plot(x, power_y_fit, color='green', label=f'Power Fit (R²={power_r2:.2f})')

            fit_results.append({
                'Combination': combo,
                'Type': 'Truck',
                'Function': 'Power',
                'a': power_params[0],
                'b': power_params[1],
                'R²': power_r2
            })
        except Exception:
            pass

        try:
            exp_params, _ = curve_fit(exp_func, x, y, p0=(1, 0.1))
            exp_y_fit = exp_func(x, *exp_params)
            exp_r2 = r2_score(y, exp_y_fit)
            axes[1].plot(x, exp_y_fit, color='orange', label=f'Exp Fit (R²={exp_r2:.2f})')

            fit_results.append({
                'Combination': combo,
                'Type': 'Truck',
                'Function': 'Exp',
                'a': exp_params[0],
                'b': exp_params[1],
                'R²': exp_r2
            })
        except Exception:
            pass

        axes[1].set_title('Truck')
        axes[1].set_xlabel('Density (veh/m)')
        axes[1].legend()

    plt.suptitle(f"Density-Speed Function\n{combo}")
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{combo['Rows']}_{combo['Cols']}_{combo['BlockLen']}_{combo['Throughput']}.png")
    plt.savefig(output_path)
    plt.close()

fit_results_df = pd.DataFrame(fit_results)
fit_results_df.to_excel(os.path.join(output_dir, 'density_speed_fit_results.xlsx'), index=False)

print("\nDone!")
