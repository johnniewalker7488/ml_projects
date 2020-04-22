# A linear regression baseline for preprocessed data

import Data_preprocessing as dp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge



model = LinearRegression()
data_prep = dp.Preprocessor()
X_train, X_test, y_train, y_test = data_prep.run_preprocessing()
#y_train = y_train.values.reshape(-1, 1)
#y_test = y_test.values.reshape(-1, 1)

model.fit(X_train, y_train)
#y_pred_tr = model.predict(X_train)
#y_pred = model.predict(X_test)


score = model.score(X_test, y_test)
print(f"The Score on the test set is {score}")
score = model.score(X_train, y_train)
print(f"The Score on the train set is {score}")

