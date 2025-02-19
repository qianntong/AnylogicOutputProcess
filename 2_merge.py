import os
import re
import pandas as pd

current_dir = r"/Users/qianqiantong/PycharmProjects/AnyLogicOutputProcess/results"
output_filename = "output/all_results.xlsx"

all_data = []

for file in os.listdir(current_dir):
    if file.startswith("intermodal_") and file.endswith("results.xlsx"):
        file_path = os.path.join(current_dir, file)

        try:
            numbers = re.findall(r'\d+', file)
            number_str = "_".join(numbers)

            xls = pd.ExcelFile(file_path)

            if 'truck' in xls.sheet_names:
                df_truck = pd.read_excel(xls, sheet_name='truck')
                if 'TruckAvgDensity(veh/m)' in df_truck.columns and 'TruckAvgSpeed(m/s)' in df_truck.columns:
                    truck_density_col = f"{number_str}_TruckAvgDensity(veh/m)"
                    truck_speed_col = f"{number_str}_TruckAvgSpeed(m/s)"

                    truck_data = df_truck[['TruckAvgDensity(veh/m)', 'TruckAvgSpeed(m/s)']].rename(columns={
                        'TruckAvgDensity(veh/m)': truck_density_col,
                        'TruckAvgSpeed(m/s)': truck_speed_col
                    })
                    all_data.append(truck_data)

            # 提取 hostler sheet
            if 'hostler' in xls.sheet_names:
                df_hostler = pd.read_excel(xls, sheet_name='hostler')
                if 'HostlerAvgDensity(veh/m)' in df_hostler.columns and 'HostlerAvgSpeed(m/s)' in df_hostler.columns:
                    hostler_density_col = f"{number_str}_HostlerAvgDensity(veh/m)"
                    hostler_speed_col = f"{number_str}_HostlerAvgSpeed(m/s)"

                    hostler_data = df_hostler[['HostlerAvgDensity(veh/m)', 'HostlerAvgSpeed(m/s)']].rename(columns={
                        'HostlerAvgDensity(veh/m)': hostler_density_col,
                        'HostlerAvgSpeed(m/s)': hostler_speed_col
                    })
                    all_data.append(hostler_data)

        except Exception as e:
            print(f"Error reading {file}: {e}")

if all_data:
    merged_df = pd.concat(all_data, axis=1)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    merged_df.to_excel(output_filename, index=False)
    print(f"Merged data saved to {output_filename}")
else:
    print("No valid data extracted.")
