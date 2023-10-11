# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13, 2023
@author: erdem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from the CSV file
df = pd.read_csv("data.csv", sep=";", header=None)

# Extract the independent variable (x) and dependent variable (y)
x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.ravel()

#%%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)  # n_estimators defines the number of trees to be used in the ensemble, while random_state ensures reproducibility.

rf.fit(x, y)

print("Price for Level 7.5: ", rf.predict([[7.5]]))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
y_head = rf.predict(x_)

#%%Plt

plt.scatter(x, y, color="red")
plt.plot(x_, y_head, color="green")
plt.xlabel("Grandstand Level \n Price for Level 7.5: {} \nPrice for Level 3.5: {}\nPrice for Level 10.5: {}".format(rf.predict([[7.5]]),rf.predict([[3.5]]),rf.predict([[10.5]])))
plt.ylabel("Price")
plt.show()