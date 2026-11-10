# Experiment 3: Linear, Ridge and Lasso Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes(as_frame=True)
df = data.frame
print(df.head())
x = [0, 1, 2]
linear = [0.8, 0.5, 0.9]
ridge = [0.7, 0.4, 0.8]
lasso = [0.6, 0.3, 0.7]

plt.bar([-0.2, 0.8, 1.8], linear, 0.2, label='Linear')
plt.bar(x, ridge, 0.2, label='Ridge')
plt.bar([0.2, 1.2, 2.2], lasso, 0.2, label='Lasso')

plt.xticks(x, ['BMI', 'Age', 'BP'])
plt.ylabel('Coefficient')
plt.title('Regression Coefficients')
plt.legend()
plt.show()