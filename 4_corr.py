import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'output/fitting_results.xlsx'
df = pd.read_excel(file_path)


df['Vehicle Type'] = df['Vehicle Type'].map({'Trucks': 1, 'Hostlers': 0})


df_trucks = df[df['Vehicle Type'] == 1]
df_hostlers = df[df['Vehicle Type'] == 0]

# Pearson Correlation Coefficient
# method = 'pearson': linear
# method = 'spearman': non-linear or sequence data
# method = 'kendall': small sample or classification data
correlation_matrix_trucks = df_trucks.corr(numeric_only=True, method = 'pearson')
correlation_matrix_hostlers = df_hostlers.corr(numeric_only=True, method = 'pearson')


print("truck correlation matrix:")
print(correlation_matrix_trucks)
print("\nhostler correlation matrix:")
print(correlation_matrix_hostlers)

plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_trucks, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Truck Correlation Matrix')
plt.show()

plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_hostlers, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Hostler Correlation Matrix')
plt.show()
