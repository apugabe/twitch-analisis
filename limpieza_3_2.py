import pandas as pd

# Cargar el dataset
df = pd.read_csv(r"C:\TwitchData\TwitchData.csv", sep=';')

# Convertir columnas con comas a números reales
columns_to_convert = ['Followers', 'Streams', 'AverageViews', 'WatchTime', 'PeakViewers', 'StreamTime']
for col in columns_to_convert:
    df[col] = df[col].replace(',', '', regex=True).astype(float)

# Mostrar tipos de datos de todas las columnas
print("Tipos de datos:")
print(df.dtypes)

# Ver cuántos valores únicos tiene la columna 'User'
print("\nValores únicos en la columna 'User':", df['User'].nunique())
