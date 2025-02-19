import os
import pandas as pd

current_dir = r"/Users/qianqiantong/PycharmProjects/AnyLogicOutputProcess/results/intermodal_layout"
output_filename = "output/same_layout.xlsx"

all_data = []

for root, dirs, files in os.walk(current_dir):
    if os.path.basename(root) == "output":
        for file in files:
            if file.endswith("results.xlsx"):
                file_path = os.path.join(root, file)

                try:
                    xls = pd.ExcelFile(file_path)

                    if 'truck' in xls.sheet_names:
                        df_truck = pd.read_excel(xls, sheet_name='truck')

                        if 'TruckAvgDensity(veh/m)' in df_truck.columns and 'TruckAvgSpeed(m/s)' in df_truck.columns:
                            base_name = file.replace("_results.xlsx", "").replace("intermodal_", "")
                            truck_density_col = f"{os.path.basename(os.path.dirname(root))}_{file}_TruckAvgDensity(veh/m)"
                            truck_speed_col = f"{os.path.basename(os.path.dirname(root))}_{file}_TruckAvgSpeed(m/s)"

                            truck_data = df_truck[['TruckAvgDensity(veh/m)', 'TruckAvgSpeed(m/s)']].rename(columns={
                                'TruckAvgDensity(veh/m)': truck_density_col,
                                'TruckAvgSpeed(m/s)': truck_speed_col
                            })
                            all_data.append(truck_data)

                    if 'hostler' in xls.sheet_names:
                        df_hostler = pd.read_excel(xls, sheet_name='hostler')
                        if 'HostlerAvgDensity(veh/m)' in df_hostler.columns and 'HostlerAvgSpeed(m/s)' in df_hostler.columns:
                            base_name = file.replace("_results.xlsx", "").replace("intermodal_", "")
                            hostler_density_col = f"{os.path.basename(os.path.dirname(root))}_{file}_HostlerAvgDensity(veh/m)"
                            hostler_speed_col = f"{os.path.basename(os.path.dirname(root))}_{file}_HostlerAvgSpeed(m/s)"

                            hostler_data = df_hostler[['HostlerAvgDensity(veh/m)', 'HostlerAvgSpeed(m/s)']].rename(columns={
                                'HostlerAvgDensity(veh/m)': hostler_density_col,
                                'HostlerAvgSpeed(m/s)': hostler_speed_col
                            })
                            all_data.append(hostler_data)

                except Exception as e:
                    print(f"Error reading {file}: {e}")

if all_data:
    merged_df = pd.concat(all_data, axis=1)
    merged_df.to_excel(output_filename, index=False)
    print(f"Merged data saved to {output_filename}")
else:
    print("No valid data extracted.")