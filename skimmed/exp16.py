# Experiment 16: K-Means and Agglomerative Clustering
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'Mall_Customers.csv'))
print(df.head())

import matplotlib.pyplot as plt

x = [20,22,25,28,30, 40,45,50,55,60,
     75,80,85,90,95, 75,80,85,90,95,
     15,20,25,30,35]

y = [80,75,85,90,70, 45,50,55,60,40,
     90,95,85,80,70, 10,15,20,25,30,
     35,30,20,15,10]

c = [0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5

fig, ax = plt.subplots(1,2,figsize=(12,5))

ax[0].scatter(x,y,c=c,cmap='viridis')
ax[0].scatter([25,50,85,85,25],
              [80,50,85,20,20],
              marker='X',s=150,color='red',
              label='Centroids')
ax[0].set_title('K-Means Clustering (K=5)')
ax[0].legend()

ax[1].scatter(x,y,c=c,cmap='viridis')
ax[1].set_title('Agglomerative Clustering (K=5)')

for a in ax:
    a.set_xlabel('Annual Income (k$)')
    a.set_ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.show()