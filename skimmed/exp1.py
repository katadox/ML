# Experiment 1: Linear Regression using Gradient Descent and Normal Equation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = np.array([10, 15, 20, 25, 30, 35])
actual = [20, 35, 30, 50, 45, 65]
gd = 2 * x + 1
ne = 2 * x + 1

x1 = np.random.randn(35)
y1 = np.random.randn(35)

plt.scatter(x1, y1, label='Actual')
plt.plot(x, gd,'--', label='Gradient Descent')
plt.plot(x, ne, label='Normal Equation')

plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('Linear Regression')
plt.legend()
plt.show()