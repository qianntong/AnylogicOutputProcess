import pandas as pd


file_path = "/Users/qianqiantong/PycharmProjects/AnylogicOutputProcess/output/all_results.xlsx"
df = pd.read_excel(file_path)
hostler_speed_cols = [col for col in df.columns if col.endswith('HostlerAvgSpeed(m/s)')]

if hostler_speed_cols:
    total_values = df[hostler_speed_cols].values.flatten()
    total_values = total_values[~pd.isna(total_values)]
    overall_avg_speed = total_values.mean()
    print(f"All 'HostlerAvgSpeed(m/s)' average is: {overall_avg_speed}")
else:
    print("Cannot find columns ended with 'HostlerAvgSpeed(m/s)'")
