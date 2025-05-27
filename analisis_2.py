
import pandas as pd
# Cargar el dataset desde la ruta donde está guardado
df = pd.read_csv(r"C:\TwitchData\TwitchData.csv", sep=';')
# Convertir las columnas que contienen números en formato texto con comas (",") a números reales
columns_to_convert = ['Followers', 'Streams', 'AverageViews', 'WatchTime', 'PeakViewers', 'StreamTime']
for col in columns_to_convert:
    df[col] = df[col].replace(',', '', regex=True).astype(float)
# Seleccionar las variables que se van a analizar
selected_columns = ['Followers', 'AverageViews', 'WatchTime', 'StreamTime', 'PeakViewers']
df_selected = df[selected_columns]
# Calcular estadísticas descriptivas de las variables seleccionadas
summary_stats = df_selected.describe().T
# Reorganizar y renombrar las columnas del resumen para facilitar la lectura
summary_stats = summary_stats[['min', '25%', '50%', 'mean', '75%', 'max']]
summary_stats.columns = ['Mínimo', '1er Cuartil', 'Mediana', 'Media', '3er Cuartil', 'Máximo']
# Mostrar el resultado final por pantalla
print(summary_stats)
