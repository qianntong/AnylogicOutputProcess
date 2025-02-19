import pandas as pd

# input
hostler_numbers = 15
loading_time_hr = 1/30

file_path = f'data/{hostler_numbers}_hostler_density.xlsx'
xlsx = pd.ExcelFile(file_path)

density_data = xlsx.parse(sheet_name=0)  # density: sheet 1
hostler_data = xlsx.parse(sheet_name=1)  # hostler: sheet 2
truck_data = xlsx.parse(sheet_name=2)    # truck:   sheet 3

#  check data
hostler_required_columns = {'start_time', 'end_time', 'HostlerDistance'}
if not hostler_required_columns.issubset(hostler_data.columns):
    raise ValueError("The Hostler data sheet must contain 'start_time', 'end_time', and 'HostlerDistance' columns.")

truck_required_columns = {'start_time', 'end_time', 'TruckDistance'}
if not truck_required_columns.issubset(truck_data.columns):
    raise ValueError("The Truck data sheet must contain 'start_time', 'end_time', and 'TruckDistance' columns.")


def calculate_total_vehicles(start_time, end_time, density_df):
    filtered_data = density_df[(density_df['Time'] >= start_time) & (density_df['Time'] <= end_time)]
    return filtered_data['Count'].sum() if not filtered_data.empty else 0


# hostler
hostler_data['HostlerTravelTime'] = hostler_data['end_time'] - hostler_data['start_time']
hostler_data['HostlerActualTravelTime'] = hostler_data['HostlerTravelTime'] - loading_time_hr
hostler_data['HostlerAvgSpeed(m/hr)'] = hostler_data['HostlerDistance'] / hostler_data['HostlerActualTravelTime']
hostler_data['HostlerAvgSpeed(m/s)'] = hostler_data['HostlerAvgSpeed(m/hr)'] / 3600
hostler_data['total_vehicles'] = hostler_data.apply(lambda row: calculate_total_vehicles(row['start_time'], row['end_time'], density_data), axis=1)
hostler_data['HostlerAvgDensity(veh/m)'] = hostler_data['total_vehicles'] / hostler_data['HostlerDistance']

# truck
truck_data['TruckTravelTime'] = truck_data['end_time'] - truck_data['start_time']
truck_data['TruckActualTravelTime'] = truck_data['TruckTravelTime'] - loading_time_hr
truck_data['TruckAvgSpeed(m/hr)'] = truck_data['TruckDistance'] / truck_data['TruckActualTravelTime']
truck_data['TruckAvgSpeed(m/s)'] = truck_data['TruckAvgSpeed(m/hr)'] / 3600
truck_data['total_vehicles'] = truck_data.apply(lambda row: calculate_total_vehicles(row['start_time'], row['end_time'], density_data), axis=1)
truck_data['TruckAvgDensity(veh/m)'] = truck_data['total_vehicles'] / truck_data['TruckDistance']

# output
output_file = f'output/{hostler_numbers}_hostler_results.xlsx'
with pd.ExcelWriter(output_file) as writer:
    hostler_data.to_excel(writer, sheet_name='hostler', index=False)
    truck_data.to_excel(writer, sheet_name='truck', index=False)

print("Done!")
print(f"Completed and saved to {output_file}.")