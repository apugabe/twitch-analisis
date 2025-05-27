
import pandas as pd

# Cargar el dataset
df = pd.read_csv(r"C:\TwitchData\TwitchData.csv", sep=';')

# Convertir columnas con separadores de miles a numéricas
columns_to_convert = ['Followers', 'Streams', 'AverageViews', 'WatchTime', 'PeakViewers', 'StreamTime']
for col in columns_to_convert:
    df[col] = df[col].replace(',', '', regex=True).astype(float)

# Verificar si hay ceros o nulos
print("Valores nulos por columna:")
print(df.isna().sum())
print("\nValores igual a 0 por columna:")
print((df[columns_to_convert] == 0).sum())

# Imputar el único valor 0 de StreamTime con la mediana
mediana_streamtime = df['StreamTime'].median()
df.loc[df['StreamTime'] == 0, 'StreamTime'] = mediana_streamtime

# Confirmar resultado
print("\nStreamTime después de imputar:")
print(df['StreamTime'].describe())
