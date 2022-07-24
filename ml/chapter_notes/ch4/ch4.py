import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Normal equation inversion
x = 2*np.random.randn(100, 1)
noise = np.random.randn(100, 1)
slope = 3
offset = 4
y = offset + slope * x + noise

x_vector = np.c_[np.ones((100, 1)), x]
x_matrix_inv = np.linalg.inv(x_vector.T.dot(x_vector))
theta_best = x_matrix_inv.dot(x_vector.T).dot(y)
y_predict = x_vector.dot(theta_best)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, y_predict)
plt.show()

# Alternatively, we can perform linear regression using sklearn
lin_reg = LinearRegression()
lin_reg.fit(x, y)
