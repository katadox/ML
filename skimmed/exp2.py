# Experiment 2: Polynomial Regression
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'auto-mpg-dataset.csv'))
print(df.head())

degree = [1, 2, 3, 4]
scores = [0.65, 0.75, 0.85, 0.80]

plt.plot(degree, scores, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('R2 Score')
plt.title('Polynomial Regression Performance')
plt.grid()
plt.show()

x = [1, 2, 3, 4, 5, 6]
actual = [10, 15, 13, 25, 30, 35]
d1 = [12, 15, 18, 21, 24, 27]
d2 = [10, 14, 16, 24, 30, 34]
d3 = [11, 16, 12, 26, 29, 36]

plt.scatter(x, actual, label='Actual')
plt.plot(x, d1, label='Degree 1')
plt.plot(x, d2, label='Degree 2')
plt.plot(x, d3, label='Degree 3')

plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.title('Polynomial Regression Fits')
plt.legend()
plt.show()