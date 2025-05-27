
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar y preparar el dataset
df = pd.read_csv(r"C:\TwitchData\TwitchData.csv", sep=';')
columns_to_convert = ['Followers', 'Streams', 'AverageViews', 'WatchTime', 'PeakViewers', 'StreamTime']
for col in columns_to_convert:
    df[col] = df[col].replace(',', '', regex=True).astype(float)

# Histograma de variables clave
df[['Followers', 'WatchTime', 'StreamTime']].hist(bins=15, figsize=(10, 6))
plt.suptitle("Distribución de variables principales")
plt.tight_layout()
plt.show()

# Diagrama de dispersión: seguidores vs. tiempo de visualización
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Followers', y='WatchTime')
plt.title("Relación entre seguidores y tiempo de visualización")
plt.xlabel("Followers")
plt.ylabel("WatchTime")
plt.show()

# Clustering y visualización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Followers', 'StreamTime', 'AverageViews']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df['cluster'] = kmeans.labels_

# Reducción a dos dimensiones para visualizar los clústeres (con PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster', palette='Set2')
plt.title("Visualización de clústeres con K-means")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()
