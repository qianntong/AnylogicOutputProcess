import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

input_file = "output/all_results.xlsx"
output_file = "output/fitting_results.xlsx"


def power_func(x, a, b):
    return a * x**b

def exp_func(x, a, b):
    return a * np.exp(b * x)


df = pd.read_excel(input_file)

column_names = df.columns
grouped_columns = [(column_names[i], column_names[i + 1]) for i in range(0, len(column_names), 2)]

fit_results = []

for x_col, y_col in grouped_columns:
    # Rows(M), Cols(N), BlockLen(n_t), Throughput(k), HostlerNum
    match = re.search(r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", x_col)
    if not match:
        continue

    row, col, block_len, throughput, hostler = map(int, match.groups())
    x = df[x_col].dropna()
    y = df[y_col].dropna()

    if len(x) == 0 or len(y) == 0:
        continue

    if "Hostler" in x_col:
        label_prefix = "Hostlers"
    elif "Truck" in x_col:
        label_prefix = "Trucks"
    else:
        continue

    power_params, exp_params = [np.nan, np.nan], [np.nan, np.nan]

    try:
        power_params, _ = curve_fit(power_func, x, y, p0=(1, -1))
    except Exception as e:
        pass

    try:
        exp_params, _ = curve_fit(exp_func, x, y, p0=(1, -0.1))
    except Exception as e:
        pass

    fit_results.append({
        "Rows": row,
        "Cols": col,
        "BlockLen": block_len,
        "Throughput": throughput,
        "HostlerNum": hostler,
        "Vehicle Type": label_prefix,
        "Power a": power_params[0],
        "Power b": power_params[1],
        "Exp a": exp_params[0],
        "Exp b": exp_params[1]
    })

fit_results_df = pd.DataFrame(fit_results)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)


print(fit_results_df)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
fit_results_df.to_excel(output_file, index=False)
print(f"Done! Results saved to {output_file}")