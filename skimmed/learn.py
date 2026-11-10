import matplotlib.pyplot as plt

x = [10, 12, 15, 40, 42, 45, 70, 72]
y = [20, 25, 22, 70, 75, 72, 30, 35]
clusters = [0, 0, 0, 1, 1, 1, 2, 2]

plt.scatter(x, y, c=clusters, cmap='viridis')
plt.scatter([12, 42, 71], [22, 72, 32],
            marker='X', s=150, label='Centroids')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering')
plt.legend()
plt.show()


x2 = [10, 12, 15, 40, 42, 45, 70, 72]
y2 = [20, 25, 22, 70, 75, 72, 30, 35]
clusters = [0, 0, 0, 1, 1, 1, 2, 2]

plt.scatter(x2, y2, c=clusters, cmap='viridis')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Agglomerative Clustering')
plt.show()