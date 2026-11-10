# Experiment 11: Support Vector Machine - Linear Kernel
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [2, 3, 2, 4, 6, 7, 6, 8]
labels = [0, 0, 0, 0, 1, 1, 1, 1]

plt.scatter(x, y, c=labels, cmap='coolwarm')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('SVM Classification')
plt.show()