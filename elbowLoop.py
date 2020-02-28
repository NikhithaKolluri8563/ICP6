import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

wcss = []
x = pd.read_csv('CC.csv', index_col=0)
x = x.apply(lambda x: x.fillna(x.mean()), axis=0)

scaler.fit(x)
x_scaler= scaler.transform(x)
x_scaled=pd.DataFrame(x_scaler, columns =x.columns)
print(x.isnull().sum())

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10,random_state=0)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
