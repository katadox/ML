# Experiment 6: Logistic Regression with and without Scaling
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

features = ['Pregnancies', 'Glucose', 'BloodPressure',
            'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigree', 'Age']

unscaled = [0.08, 0.01, 0, 0, 0, 0, 0, 0]
scaled = [0.18, 1.13, -0.19, -0.02, -0.10, 0.81, 0.14, 0.42]

plt.bar(features, unscaled, color='lightcoral',
        label='Unscaled Weights')
plt.bar(features, scaled, color='blue', alpha=0.5,
        label='Scaled Weights')

plt.axhline(0, color='black', linestyle=':')
plt.xticks(rotation=30)
plt.ylabel('Weight Parameter Coefficient Value')
plt.title('Weight Coefficients: Unscaled vs. Standardized Features')
plt.legend()
plt.tight_layout()
plt.show()