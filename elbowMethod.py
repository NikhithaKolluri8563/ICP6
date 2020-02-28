
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

x = pd.read_csv('CC.csv', index_col=0)
# x = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
x=x.apply(lambda x: x.fillna(x.mean()),axis=0)

scaler.fit(x)
x_scaler= scaler.transform(x)
x_scaled=pd.DataFrame(x_scaler, columns =x.columns)
print(x.isnull().sum())


##building the model
from sklearn.cluster import KMeans
nclusters = 4 # this is the k in kmeans
seed=0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(x_scaled) # predict the cluster for each data pointy_cluster_kmeans=km.predict(X_scaler)


# predict the cluster for each data point
y_cluster_kmeans = km.predict(x_scaled)
from sklearn import metrics
score = metrics.silhouette_score(x_scaled, y_cluster_kmeans)
print(score)