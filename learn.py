import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from pathlib import Path

basedir=Path(__file__).resolve().parent
csvpath=basedir/'data'/'auto-mpg-dataset.csv'
df=pd.read_csv(csvpath)

print(df.head())
df=df.dropna(subset=['displacement','mpg'])
x=df[['displacement']].values
y=df['mpg'].values

x_trian,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

linmod=LinearRegression()
linmod.fit(x_trian,y_train)


