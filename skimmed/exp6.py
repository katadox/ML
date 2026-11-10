# Experiment 6: Logistic Regression with and without Scaling
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

features = ['Age', 'BMI', 'Glucose']
unscaled = [0.5, 0.8, 0.9]
scaled = [0.3, 0.6, 0.7]

plt.bar(features, unscaled, alpha=0.6, label='Unscaled')
plt.bar(features, scaled, alpha=0.6, label='Scaled')

plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.title('Logistic Regression')
plt.legend()
plt.show()