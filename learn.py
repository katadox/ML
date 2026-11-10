import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report

mnist=fetch_openml('mnist_784',version=1,as_frame=False, parser='liac-arff')
x,y=mnist.data.astype(np.float32)/255.0,mnist.target.astype(int)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)
print("Dataset processing complete.")
history_loss={}
test_accuracies={}
activations={'logistic','tanh','relu'}
for act in activations:
    mlp=MLPClassifier(hidden_layer_sizes=(64,32),activation=act,max_iter=25,random_state=42)
    mlp.fit(x_train,y_train)
    history_loss[act]=mlp.loss_curve_
    test_accuracies[act]=accuracy_score(y_test,mlp.predict(x_test))
print(test_accuracies)

plt.figure(figsize=(10,5))
for act in activations:
    print(f"accuracy for {'sigmoid' if act =='logistic' else act}:"f"{test_accuracies[act]:4f}")
    plt.plot(history_loss[act],label=act)

plt.title('loss convergence comparison')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend()
plt.show()