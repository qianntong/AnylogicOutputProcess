import pandas as pd
import os
import re

# Constants
loading_time_hr = 1 / 30
hostler_numbers_list = [5, 10, 15]

# Input and output directories
input_dir = 'data'
output_dir = 'results'
layout_file = 'data/layout.xlsx'
os.makedirs(output_dir, exist_ok=True)


def calculate_total_vehicles(start_time, end_time, density_df):
    filtered_data = density_df[(density_df['Time'] >= start_time) & (density_df['Time'] <= end_time)]
    return filtered_data['Count'].sum() if not filtered_data.empty else 0

# Read layout.xlsx file
try:
    layout_df = pd.read_excel(layout_file)
    required_columns = {'M', 'N', 'n_t', 'k', 'A', 'B'}
    if not required_columns.issubset(layout_df.columns):
        raise ValueError(f"Layout file {layout_file} must contain columns: {required_columns}")
except Exception as e:
    raise FileNotFoundError(f"Error reading layout file {layout_file}: {str(e)}")



for folder_name in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder_name)

    # Check if it's a directory and matches the pattern
    if os.path.isdir(folder_path) and folder_name.startswith('intermodal_'):
        # Extract M, N, n_t, and k from folder name
        match = re.search(r'intermodal_(\d+)_(\d+)_(\d+)_(\d+)', folder_name)
        if match:
            M, N, n_t, k = map(int, match.groups())

            # Find A and B from layout excel
            param_row = layout_df[(layout_df['M'] == M) &
                                  (layout_df['N'] == N) &
                                  (layout_df['n_t'] == n_t) &
                                  (layout_df['k'] == k)]
            if param_row.empty:
                print(f"No matching A and B found for M={M}, N={N}, n_t={n_t}, k={k}, skipping folder.")
                continue

            A, B = param_row.iloc[0]['A'], param_row.iloc[0]['B']
            total_lane_length = A * (N + 1) + B * (M + 1)
            print(f"\nProcessing folder: {folder_name}, M: {M}, N: {N}, n_t: {n_t}, k: {k}, A: {A}, B: {B}, Total lane length: {total_lane_length}")

            # Loop through each hostler number
            for hostler_numbers in hostler_numbers_list:
                file_path = os.path.join(folder_path, f'{hostler_numbers}_hostler_density.xlsx')

                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}, skipping.")
                    continue

                print(f"Processing file: {file_path}")

                try:
                    # Load data
                    xlsx = pd.ExcelFile(file_path)
                    density_data = xlsx.parse(sheet_name=0)  # Density: sheet 1
                    hostler_data = xlsx.parse(sheet_name=1)  # Hostler: sheet 2
                    truck_data = xlsx.parse(sheet_name=2)  # Truck: sheet 3

                    # Validate columns
                    hostler_required_columns = {'start_time', 'end_time', 'HostlerDistance'}
                    truck_required_columns = {'start_time', 'end_time', 'TruckDistance'}

                    if not hostler_required_columns.issubset(hostler_data.columns):
                        raise ValueError(f"Hostler data in {file_path} must contain 'start_time', 'end_time', and 'HostlerDistance' columns.")
                    if not truck_required_columns.issubset(truck_data.columns):
                        raise ValueError(f"Truck data in {file_path} must contain 'start_time', 'end_time', and 'TruckDistance' columns.")

                    # Hostler calculations
                    hostler_data['HostlerTravelTime'] = hostler_data['end_time'] - hostler_data['start_time']
                    hostler_data['HostlerActualTravelTime'] = hostler_data['HostlerTravelTime'] - loading_time_hr
                    hostler_data['HostlerAvgSpeed(m/hr)'] = hostler_data['HostlerDistance'] / hostler_data['HostlerActualTravelTime']
                    hostler_data['HostlerAvgSpeed(m/s)'] = hostler_data['HostlerAvgSpeed(m/hr)'] / 3600
                    hostler_data['total_vehicles'] = hostler_data.apply(lambda row: calculate_total_vehicles(row['start_time'], row['end_time'], density_data), axis=1)
                    hostler_data['HostlerAvgDensity(veh/m)'] = hostler_data['total_vehicles'] / total_lane_length

                    # Truck calculations
                    truck_data['TruckTravelTime'] = truck_data['end_time'] - truck_data['start_time']
                    truck_data['TruckActualTravelTime'] = truck_data['TruckTravelTime'] - loading_time_hr
                    truck_data['TruckAvgSpeed(m/hr)'] = truck_data['TruckDistance'] / truck_data['TruckActualTravelTime']
                    truck_data['TruckAvgSpeed(m/s)'] = truck_data['TruckAvgSpeed(m/hr)'] / 3600
                    truck_data['total_vehicles'] = truck_data.apply(lambda row: calculate_total_vehicles(row['start_time'], row['end_time'], density_data), axis=1)
                    truck_data['TruckAvgDensity(veh/m)'] = truck_data['total_vehicles'] / total_lane_length

                    # Output
                    output_file = os.path.join(output_dir, f'{folder_name}_{hostler_numbers}_results.xlsx')
                    with pd.ExcelWriter(output_file) as writer:
                        hostler_data.to_excel(writer, sheet_name='hostler', index=False)
                        truck_data.to_excel(writer, sheet_name='truck', index=False)

                    print(f"Completed and saved to {output_file}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

        else:
            print(f"Folder name format incorrect: {folder_name}, skipping folder.")
    else:
        print(f"Skipping non-target folder: {folder_name}")

print("\nAll processing done!")