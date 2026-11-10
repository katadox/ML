import numpy as np
import pandas as pd #not needed
import matplotlib.pyplot as plt
import time #not needed
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report  #classification_report is not needed
# Load MNIST data. The default parser is currently incompatible with the installed
# sklearn/pandas combination on this machine, so use the ARFF parser that still
# handles OpenML's mnist_784 dataset correctly.
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff') #why as_frame=False? because we want to get the data as a numpy array instead of a pandas dataframe. This is because the MLPClassifier expects the input data to be in the form of a numpy array.
X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int) # Normalize the pixel values to the range [0, 1] and convert the target labels to integers.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print("Dataset processing complete.")

history_loss = {}
test_accuracies = {}
activations = ['logistic', 'tanh', 'relu']
for act in activations:
    print(f"Training network using activation: {act}")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation=act, max_iter=25,random_state=42)  #each activation function will be tested with a network of 2 hidden layers, the first with 128 neurons and the second with 64 neurons. The maximum number of iterations for training is set to 25.
    mlp.fit(X_train, y_train) #why fit? The fit method is used to train the MLPClassifier on the training data (X_train, y_train). It adjusts the weights of the neural network based on the input data and the corresponding labels to minimize the loss function.
    history_loss[act] = mlp.loss_curve_ #it stores the loss values at each iteration of training for the current activation function in the history_loss dictionary. This allows us to visualize how the loss changes over time for different activation functions.
    test_accuracies[act] = accuracy_score(y_test, mlp.predict(X_test))
print(test_accuracies)

plt.figure(figsize=(10, 5))
for act in activations:
    print(f"Accuracy for {'sigmoid' if act == 'logistic' else act}: {test_accuracies[act]:.4f}")
    plt.plot(history_loss[act], label=act)
plt.title('Loss Convergence Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.show()
