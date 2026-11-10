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
import matplotlib.pyplot as plt
import numpy as np

x = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']

a = [2,-12,26,17,-44,25,8,13,35,3]
b = [2,-10,24,15,-5,-4,-9,7,19,4]
c = [0,-8,26,15,-5,0,-11,0,22,2]

w = 0.25
p = np.arange(len(x))

plt.bar(p-w, a, w, label='Linear')
plt.bar(p, b, w, label='Ridge')
plt.bar(p+w, c, w, label='Lasso')

plt.xticks(p, x, rotation=45)
plt.axhline(0, color='black')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Comparison: Linear vs Ridge vs Lasso')
plt.legend()
plt.tight_layout()
plt.show()