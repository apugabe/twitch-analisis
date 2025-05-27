import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Cargar el dataset
df = pd.read_csv(r"C:\TwitchData\TwitchData.csv", sep=';')
columns_to_convert = ['Followers', 'Streams', 'AverageViews', 'WatchTime', 'PeakViewers', 'StreamTime']
for col in columns_to_convert:
    df[col] = df[col].replace(',', '', regex=True).astype(float)

# Modelo supervisado: regresión lineal
X = df[['Followers', 'StreamTime']]
y = df['WatchTime']
modelo = LinearRegression().fit(X, y)
y_pred = modelo.predict(X)
r2 = r2_score(y, y_pred)
coefs = dict(zip(X.columns, modelo.coef_))
intercept = modelo.intercept_

print("Regresión lineal para predecir WatchTime:")
print(f"R²: {r2:.2f}")
print("Coeficientes:")
for var, coef in coefs.items():
    print(f"{var}: {coef:.2f}")
print(f"Intercepto: {intercept:.2f}")

# Modelo no supervisado: K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Followers', 'StreamTime', 'AverageViews']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df['cluster'] = kmeans.labels_

print("\nDistribución de canales por grupo (K-means):")
print(df['cluster'].value_counts().sort_index())
