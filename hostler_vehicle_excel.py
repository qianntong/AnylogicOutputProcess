import pandas as pd

def calculate_average_from_excel(input_file, intervals_file, output_file):
    data = pd.read_excel(input_file)

    if not {'Time', 'Count'}.issubset(data.columns):
        raise ValueError("The main Excel file must contain 'Time' and 'Count' columns.")

    intervals = pd.read_excel(intervals_file)

    # Ensure the columns are named correctly in the intervals file
    if not {'start_time', 'end_time'}.issubset(intervals.columns):
        raise ValueError("The intervals Excel file must contain 'start_time' and 'end_time' columns.")

    # # Sort the data by time (if not already sorted)
    # data = data.sort_values(by='Time').reset_index(drop=True)

    results = []

    # Iterate over each row in the intervals file
    for _, row in intervals.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']

        # Find the closest time greater than or equal to time interval
        start_index = data[data['Time'] >= start_time].index.min()
        end_index = data[data['Time'] <= end_time].index.max()

        # Check if valid indices were found
        if start_index is None or end_index is None or start_index > end_index:
            total_veh = None  # Assign None if the interval is invalid
        else:
            print(f"start_index: {start_index}, type: {type(start_index)}")
            print(f"end_index: {end_index}, type: {type(end_index)}")
            filtered_data = data.iloc[start_index:end_index + 1]
            total_veh = filtered_data['Count'].sum()

        # Append the result to the list
        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'total_vehicles': total_veh
        })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Write the results to an Excel file
    results_df.to_excel(output_file, index=False)

# Example usage
input_file = 'data/density.xlsx'
intervals_file = 'data/interval.xlsx'
output_file = 'data/vehicle_density_output.xlsx'


calculate_average_from_excel(input_file, intervals_file, output_file)

# try:
#     calculate_average_from_excel(input_file, intervals_file, output_file)
#     print(f"The averages have been successfully calculated and saved to {output_file}.")
# except Exception as e:
#     print(f"Error: {e}")