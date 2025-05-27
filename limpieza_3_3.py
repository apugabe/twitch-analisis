import pandas as pd

# Cargar y preparar el dataset
df = pd.read_csv(r"C:\TwitchData\TwitchData.csv", sep=';')
columns_to_convert = ['Followers', 'Streams', 'AverageViews', 'WatchTime', 'PeakViewers', 'StreamTime']
for col in columns_to_convert:
    df[col] = df[col].replace(',', '', regex=True).astype(float)

# Detección de valores extremos usando el criterio IQR
outlier_summary = {}
for col in columns_to_convert:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)

# Mostrar cantidad de outliers por variable
print("Número de valores extremos detectados por variable:")
for var, count in outlier_summary.items():
    print(f"{var}: {count}")
