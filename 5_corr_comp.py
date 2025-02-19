import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_path = 'output/fitting_results.xlsx'
df = pd.read_excel(file_path)

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

independent_vars = ['Rows', 'Cols', 'BlockLen', 'Throughput', 'HostlerNum']
dependent_vars = ['Power a', 'Power b', 'Exp a', 'Exp b']

df['Vehicle Type'] = df['Vehicle Type'].map({'Trucks': 1, 'Hostlers': 0})
df_trucks = df[df['Vehicle Type'] == 1]
df_hostlers = df[df['Vehicle Type'] == 0]

correlation_results_trucks = pd.DataFrame()
correlation_results_hostlers = pd.DataFrame()

for dep_var in dependent_vars:
    correlations_trucks = []
    for ind_var in independent_vars:
        corr_value = df_trucks[[ind_var, dep_var]].corr(method='pearson').iloc[0, 1]
        correlations_trucks.append(corr_value)

    correlations_hostlers = []
    for ind_var in independent_vars:
        corr_value = df_hostlers[[ind_var, dep_var]].corr(method='pearson').iloc[0, 1]
        correlations_hostlers.append(corr_value)

    temp_df_trucks = pd.DataFrame({
        'Independent Variable': independent_vars,
        'Correlation': correlations_trucks,
        'Dependent Variable': dep_var,
        'Vehicle Type': 'Trucks'
    })
    temp_df_hostlers = pd.DataFrame({
        'Independent Variable': independent_vars,
        'Correlation': correlations_hostlers,
        'Dependent Variable': dep_var,
        'Vehicle Type': 'Hostlers'
    })

    correlation_results_trucks = pd.concat([correlation_results_trucks, temp_df_trucks], ignore_index=True)
    correlation_results_hostlers = pd.concat([correlation_results_hostlers, temp_df_hostlers], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    sns.barplot(ax=axes[0], x='Independent Variable', y='Correlation', data=temp_df_hostlers, palette='coolwarm')
    axes[0].set_title(f'Hostlers - {dep_var}')
    axes[0].set_ylim(-1, 1)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].set_xticklabels(independent_vars, rotation=45)

    sns.barplot(ax=axes[1], x='Independent Variable', y='Correlation', data=temp_df_trucks, palette='coolwarm')
    axes[1].set_title(f'Trucks - {dep_var}')
    axes[1].set_ylim(-1, 1)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_xticklabels(independent_vars, rotation=45)

    plt.suptitle(f'Correlation Analysis: {dep_var}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_{dep_var}.png'))
    plt.show()

correlation_results_trucks.to_excel(os.path.join(output_dir, 'correlation_results_trucks.xlsx'), index=False)
correlation_results_hostlers.to_excel(os.path.join(output_dir, 'correlation_results_hostlers.xlsx'), index=False)

print("\nDone! Correlation results and figures saved in the 'output' folder.")
