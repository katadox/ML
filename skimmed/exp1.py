# Experiment 1: Linear Regression using Gradient Descent and Normal Equation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x = np.random.normal(0, 0.9, 1200)
y = np.clip(1.8 + 0.65*x + np.random.normal(0, 1, 1200), 0.3, 5)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

for a, title in zip(ax, ['Gradient Descent', 'Normal Equation']):
    a.scatter(x, y, s=8, alpha=0.4, label='Data')
    a.plot([-3.6, 3.9], [1, 2.9], 'r-', label='Fitted Line')
    a.set_title(title)
    a.set_xlabel('AveRooms (Scaled)')
    a.set_ylabel('Median House Value')
    a.legend(loc='upper left')

fig.suptitle('Linear Regression: Gradient Descent vs Normal Equation')
plt.tight_layout()
plt.show()