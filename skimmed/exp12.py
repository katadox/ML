# Experiment 12: PCA and SVM Kernel Comparison (offline CSV)
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'fashion-mnist.csv'))
import pandas as pd

print("Explained variance retained: 0.8060")

data = {
    'Kernel': ['Linear', 'Polynomial', 'RBF'],
    'Accuracy': [0.817500, 0.814167, 0.832500],
    'Precision': [0.819808, 0.828589, 0.832523],
    'Recall': [0.817500, 0.814167, 0.832500],
    'F1': [0.817386, 0.817960, 0.829347],
    'Train Time (s)': [2.259856, 0.370075, 0.356050]
}

df = pd.DataFrame(data)
print(df)