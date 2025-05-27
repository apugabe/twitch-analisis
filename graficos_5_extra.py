import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Cargar y preparar el dataset
df = pd.read_csv(r"C:\TwitchData\TwitchData.csv", sep=';')
columns_to_convert = ['Followers', 'Streams', 'AverageViews', 'WatchTime', 'PeakViewers', 'StreamTime']
for col in columns_to_convert:
    df[col] = df[col].replace(',', '', regex=True).astype(float)

# Diagrama de dispersión: Followers vs. WatchTime
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Followers', y='WatchTime')
plt.title("Relación entre Followers y WatchTime")
plt.xlabel("Followers")
plt.ylabel("WatchTime")
plt.tight_layout()
plt.savefig(r"C:\TwitchData\scatter_followers_watchtime.png")
plt.close()

# K-means + PCA para visualización de clústeres
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Followers', 'StreamTime', 'AverageViews']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df['cluster'] = kmeans.labels_

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster', palette='Set2')
plt.title("Visualización de clústeres (K-means + PCA)")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend(title="Grupo")
plt.tight_layout()
plt.savefig(r"C:\TwitchData\clusters_pca.png")
plt.close()
