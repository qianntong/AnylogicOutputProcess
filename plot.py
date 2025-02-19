# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# file_path = "data/all_data.xlsx"
# df = pd.read_excel(file_path, sheet_name='sheet 1')
#
#
# hostler_groups = sorted(set(col.split("_")[0] for col in df.columns if "Hostler" in col))
# truck_groups = sorted(set(col.split("_")[0] for col in df.columns if "Truck" in col))
# print("Available Hostler Groups:", hostler_groups)
# print("Available Truck Groups:", truck_groups)
# selected_hostlers = input("Enter Hostler groups to plot (comma-separated): ").split(',')
# selected_trucks = input("Enter Truck groups to plot (comma-separated): ").split(',')
# selected_hostlers = [group.strip() for group in selected_hostlers if group.strip() in hostler_groups]
# selected_trucks = [group.strip() for group in selected_trucks if group.strip() in truck_groups]
#
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
#
# # Hostler
# if selected_hostlers:
#     fig1, ax1 = plt.subplots(figsize=(8, 6))
#     for i, group in enumerate(selected_hostlers):
#         density_col = f"{group}_HostlerAvgDensity(veh/m)"
#         speed_col = f"{group}_HostlerAvgSpeed(m/s)"
#         if density_col in df.columns and speed_col in df.columns:
#             ax1.scatter(df[density_col], df[speed_col], label=f"{group} hostlers", color=colors[i % len(colors)], alpha=0.7)
#
#     ax1.set_xlabel("Hostler Avg Density (veh/m)")
#     ax1.set_ylabel("Hostler Avg Speed (m/s)")
#     ax1.set_title("Hostler Speed vs Density")
#     ax1.legend()
#     ax1.grid(True)
#
# # Truck
# if selected_trucks:
#     fig2, ax2 = plt.subplots(figsize=(8, 6))
#     for i, group in enumerate(selected_trucks):
#         density_col = f"{group}_TruckAvgDensity(veh/m)"
#         speed_col = f"{group}_TruckAvgSpeed(m/s)"
#         if density_col in df.columns and speed_col in df.columns:
#             ax2.scatter(df[density_col], df[speed_col], label=f"{group} trucks", color=colors[i % len(colors)], alpha=0.7)
#
#     ax2.set_xlabel("Truck Avg Density (veh/m)")
#     ax2.set_ylabel("Truck Avg Speed (m/s)")
#     ax2.set_title("Truck Speed vs Density")
#     ax2.legend()
#     ax2.grid(True)
#
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


file_path = "data/all_data.xlsx"
df = pd.read_excel(file_path, sheet_name='sheet 1')

# 获取所有可用的 Hostler 和 Truck 组
hostler_groups = sorted(set(col.split("_")[0] for col in df.columns if "Hostler" in col))
truck_groups = sorted(set(col.split("_")[0] for col in df.columns if "Truck" in col))

# 显示可选项
print("Available Hostler Groups:", hostler_groups)
print("Available Truck Groups:", truck_groups)

# 让用户选择要绘制的组
selected_hostlers = input("Enter Hostler groups to plot (comma-separated): ").split(',')
selected_trucks = input("Enter Truck groups to plot (comma-separated): ").split(',')

# 清理用户输入并转换为列表
selected_hostlers = [group.strip() for group in selected_hostlers if group.strip() in hostler_groups]
selected_trucks = [group.strip() for group in selected_trucks if group.strip() in truck_groups]


# 定义拟合函数
def power_func(x, a, b):
    return a * x ** b


def exp_func(x, a, b):
    return a * np.exp(b * x)


# 拟合并绘制函数
def plot_with_fit(ax, x, y, group_label, color):
    # 绘制散点图
    ax.scatter(x, y, label=f"{group_label}", color=color, alpha=0.7)

    # 尝试幂函数拟合
    try:
        params, _ = curve_fit(power_func, x, y, p0=(1, -1))
        y_fit = power_func(x, *params)
        r2 = r2_score(y, y_fit)
        ax.plot(x, y_fit, color=color, linestyle='--',
                label=f"Power Fit: y={params[0]:.2f}*x^{params[1]:.2f}, R²={r2:.2f}")
    except:
        print(f"Power fit failed for {group_label}")

    # 尝试指数函数拟合
    try:
        params, _ = curve_fit(exp_func, x, y, p0=(1, -0.1))
        y_fit = exp_func(x, *params)
        r2 = r2_score(y, y_fit)
        ax.plot(x, y_fit, color=color, linestyle=':',
                label=f"Exp Fit: y={params[0]:.2f}*e^({params[1]:.2f}x), R²={r2:.2f}")
    except:
        print(f"Exponential fit failed for {group_label}")


# 颜色映射
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

# 绘制 Hostler 图
if selected_hostlers:
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, group in enumerate(selected_hostlers):
        density_col = f"{group}_HostlerAvgDensity(veh/m)"
        speed_col = f"{group}_HostlerAvgSpeed(m/s)"
        if density_col in df.columns and speed_col in df.columns:
            x = df[density_col].dropna()
            y = df[speed_col].dropna()
            plot_with_fit(ax1, x, y, f"{group} hostlers", colors[i % len(colors)])

    ax1.set_xlabel("Hostler Avg Density (veh/m)")
    ax1.set_ylabel("Hostler Avg Speed (m/s)")
    ax1.set_title("Hostler Speed vs Density")
    ax1.legend()
    ax1.grid(True)

# 绘制 Truck 图
if selected_trucks:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, group in enumerate(selected_trucks):
        density_col = f"{group}_TruckAvgDensity(veh/m)"
        speed_col = f"{group}_TruckAvgSpeed(m/s)"
        if density_col in df.columns and speed_col in df.columns:
            x = df[density_col].dropna()
            y = df[speed_col].dropna()
            plot_with_fit(ax2, x, y, f"{group} trucks", colors[i % len(colors)])

    ax2.set_xlabel("Truck Avg Density (veh/m)")
    ax2.set_ylabel("Truck Avg Speed (m/s)")
    ax2.set_title("Truck Speed vs Density")
    ax2.legend()
    ax2.grid(True)

# 显示图形
plt.show()
