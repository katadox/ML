import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print(df.describe())

# Select single feature
X = df[['AveRooms']].values
y = df['MedHouseVal'].values
# Clip outliers (optional, for better visualization)
mask = (X.flatten() < 10)
X, y = X[mask], y[mask]
# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42) # why 42? The random_state parameter is set to 42 to ensure reproducibility of the train-test split. By using a fixed random seed, the same split will be generated each time the code is run, allowing for consistent results and easier debugging.

# Feature scaling
scaler = StandardScaler() # why we use this? StandardScaler is used to standardize the features by removing the mean and scaling to unit variance. This is important for gradient descent optimization, as it helps the algorithm converge faster and prevents features with larger scales from dominating the learning process.
X_train_sc = scaler.fit_transform(X_train) #fit: Calculates the mean and standard deviation from training data, transform: Uses those values to scale the training data.
X_test_sc = scaler.transform(X_test) #Scales the testing data using the same mean and standard deviation.

def gradient_descent(X, y, lr=0.01, epochs=1000): #what is epochs? In the context of machine learning, an epoch refers to one complete pass through the entire training dataset. During each epoch, the model's parameters are updated based on the computed gradients from the loss function. Multiple epochs allow the model to learn and converge towards optimal parameters over time.
    m = len(y)
    theta0, theta1 = 0.0, 0.0
    for _ in range(epochs): #_ is a common convention in Python to indicate that the loop variable is not going to be used within the loop body. It serves as a placeholder, signaling that the loop will iterate a certain number of times (in this case, 'epochs'), but the specific iteration count is irrelevant to the logic inside the loop.
        y_p = theta0 + theta1 * X.flatten()
        error = y_p - y
        theta0 -= lr * (1/m) * np.sum(error)
        theta1 -= lr * (1/m) * np.sum(error * X.flatten()) #flatten() is used to convert the 2D array X into a 1D array, allowing for element-wise multiplication with the error array. This is necessary because the error array is also 1D, and we want to compute the gradient for theta1 based on the relationship between the feature values and the prediction errors.
    return theta0, theta1
t0_gd, t1_gd = gradient_descent(X_train_sc, y_train, lr=0.01, epochs=1000)
print(f'GD => theta0={t0_gd:.4f}, theta1={t1_gd:.4f}')

def normal_equation(X, y):
    # Add bias column of ones
    X_b = np.c_[np.ones(len(X)), X] #np.ones(len(x)) and np.c_ are used to create a new array X_b that includes a bias term (intercept) for linear regression. np.ones(len(X)) generates a column vector of ones with the same number of rows as X, representing the bias term. np.c_ is used to concatenate this column of ones with the original feature matrix X, resulting in a new matrix X_b that has an additional column for the bias term. This allows the linear regression model to learn both the slope (theta1) and the intercept (theta0) during training.
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y #X_b,@, and np.linalg.pinv are used to compute the parameters (theta) of the linear regression model using the normal equation. The normal equation is a closed-form solution that minimizes the mean squared error between the predicted and actual values. The expression X_b.T @ X_b computes the dot product of the transpose of X_b and X_b, which is part of the normal equation formula. np.linalg.pinv calculates the pseudo-inverse of this matrix, allowing for a solution even when the matrix is not invertible. Finally, this pseudo-inverse is multiplied by X_b.T @ y to obtain the optimal parameters theta that minimize the error.
    return theta[0], theta[1]
t0_ne, t1_ne = normal_equation(X_train_sc, y_train)
print(f'NE => theta0={t0_ne:.4f}, theta1={t1_ne:.4f}')

for name, t0, t1 in [('Gradient Descent', t0_gd, t1_gd),
('Normal Equation', t0_ne, t1_ne)]: #this loop run twice, once for each method (Gradient Descent and Normal Equation). In each iteration, it calculates the predicted values (y_pred) using the corresponding theta0 and theta1 parameters obtained from the respective method. It then computes the Mean Squared Error (MSE) and R-squared (R2) score for the predictions compared to the actual test values (y_test). Finally, it prints out the performance metrics for each method, allowing for a comparison of their effectiveness in fitting the linear regression model to the data.
    y_pred = t0 + t1 * X_test_sc.flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name:20s} MSE={mse:.4f} RMSE={mse**0.5:.4f} R2={r2:.4f}')



#need to understand this part
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (name, t0, t1) in zip(axes, [
    ('Gradient Descent', t0_gd, t1_gd),
    ('Normal Equation', t0_ne, t1_ne)]):
    ax.scatter(X_test_sc, y_test, alpha=0.4, s=8, color='steelblue', label='Data')
    xs = np.linspace(X_test_sc.min(), X_test_sc.max(), 200)
    ax.plot(xs, t0 + t1*xs, color='crimson', lw=2, label='Fitted Line')
    ax.set_title(name)
    ax.set_xlabel('AveRooms (Scaled)')
    ax.set_ylabel('Median House Value')
    ax.legend()
plt.suptitle('Linear Regression: Gradient Descent vs Normal Equation')
plt.tight_layout()
plt.show()