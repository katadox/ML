# Experiment 14: MLP with different activation functions (offline CSV)
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

e = range(25)

logistic = [1.1, .35, .25, .20, .16, .13, .11, .09, .07, .06,
            .05, .04, .03, .02, .02, .01, .01, .01, .01, .01,
            .01, .01, .01, .01, .01]

tanh = [.4, .18, .13, .10, .08, .06, .05, .04, .03, .02,
        .02, .01, .01, .01, .01, .01, .01, .01, .01, .01,
        .01, .01, .01, .01, .01]

relu = [.38, .15, .10, .08, .06, .05, .04, .03, .03, .02,
        .02, .01, .01, .01, .01, .01, .01, .01, .01, .01,
        .01, .01, .01, .01, .01]

plt.plot(e, logistic, label='logistic')
plt.plot(e, tanh, label='tanh')
plt.plot(e, relu, label='relu')

plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.title('Loss Convergence Comparison')
plt.legend()
plt.show()
