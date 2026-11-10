# Experiment 8: K-Nearest Neighbours
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

#df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'fashion-mnist.csv'))

k = [1, 3, 5, 11]
accuracy = [0.801, 0.804, 0.811, 0.803]
time = [0.508, 0.400, 0.444, 0.457]

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(k, accuracy, 'o-', color='tab:blue')
ax1.set_xlabel('Hyperparameter Value (K)')
ax1.set_ylabel('Test Accuracy Score', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(k, time, 's--', color='red')
ax2.set_ylabel('Inference Delay Time (Seconds)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('KNN Parameter Grid Search Analysis: Accuracy vs. Efficiency')
plt.show()