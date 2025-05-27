import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar y preparar el dataset
df = pd.read_csv(r"C:\TwitchData\TwitchData.csv", sep=';')
columns_to_convert = ['Followers', 'Streams', 'AverageViews', 'WatchTime', 'PeakViewers', 'StreamTime']
for col in columns_to_convert:
    df[col] = df[col].replace(',', '', regex=True).astype(float)

# K-means clustering (grupos ya generados)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Followers', 'StreamTime', 'AverageViews']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df['cluster'] = kmeans.labels_

# Comparar WatchTime entre dos grupos
grupo_0 = df[df['cluster'] == 0]['WatchTime']
grupo_1 = df[df['cluster'] == 1]['WatchTime']

# Comprobar normalidad y homocedasticidad
print("Shapiro grupo 0:", shapiro(grupo_0))
print("Shapiro grupo 1:", shapiro(grupo_1))
print("Levene (igualdad de varianzas):", levene(grupo_0, grupo_1))

# Prueba t (no asume varianzas iguales)
t_test = ttest_ind(grupo_0, grupo_1, equal_var=False)
print("t =", t_test.statistic)
print("p =", t_test.pvalue)
