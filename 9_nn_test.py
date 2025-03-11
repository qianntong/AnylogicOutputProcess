import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# ✅ 神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.activation = nn.Sigmoid()  # 使用 Sigmoid 激活函数
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x


# ✅ 评估函数
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - 1 - 1)  # p = 1 (Density)
    return mse, rmse, mae, mape, r2, adj_r2


# ✅ 文件路径
input_file = 'output/all_results.xlsx'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(input_file)

# test set {'Rows': 2, 'Cols': 2, 'BlockLen': 19, 'Throughput': 150}
combo_pattern = "2_2_19_150"

df_hostler_list = []
df_truck_list = []

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
    print("Warning: No data found for the specified combination")
else:
    plt.figure(figsize=(8, 6))

    def train_neural_net(x, y, color, label):
        x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        model = NeuralNet()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # 训练模型
        epochs = 5000
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred = model(x_tensor).numpy().flatten()
            mse, rmse, mae, mape, r2, adj_r2 = evaluate_model(y, y_pred)

        print(f"\n{label} Results:")
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}")

        sns.scatterplot(x=x, y=y, color=color, label=label + f" Data (R²={r2:.2f})")

        x_fit = np.linspace(min(x), max(x), 100)
        x_fit_tensor = torch.tensor(x_fit, dtype=torch.float32).view(-1, 1)
        y_fit = model(x_fit_tensor).detach().numpy().flatten()

        line_color = "green" if label == "Hostler" else "yellow"
        plt.plot(x_fit, y_fit, color=line_color, linewidth=2, label=f'{label} Fit (RMSE={rmse:.2f})')

    if not df_hostler.empty:
        x_hostler = df_hostler['Density (veh/m)'].values
        y_hostler = df_hostler['Speed (m/s)'].values
        train_neural_net(x_hostler, y_hostler, color='blue', label='Hostler')

    # ✅ Truck
    if not df_truck.empty:
        x_truck = df_truck['Density (veh/m)'].values
        y_truck = df_truck['Speed (m/s)'].values
        train_neural_net(x_truck, y_truck, color='red', label='Truck')

    plt.title("Density-Speed Function with Sigmoid (Test Set: 2_2_19_150)")
    plt.xlabel('Density (veh/m)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "Sigmoid_Test_2_2_19_150.png")
    plt.savefig(output_path)
    plt.close()

print("\nDone!")
