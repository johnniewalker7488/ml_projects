# A linear regression baseline for preprocessed data

import Data_preprocessing as dp

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge


def lin_reg():
    filename = 'Git/ml_projects/Regression/Used_cars_price_prediction/true_car_listings.csv'
    model = LinearRegression()
    data_prep = dp.Preprocessor()
    df_clean = data_prep.clean_data(filename)
    X_train, X_test, y_train, y_test = data_prep.split_data(df_clean, test_size=0.2)
    X_train, X_test = data_prep.scale_numeric(X_train, X_test)
    X_train, X_test = data_prep.one_hot(X_train, X_test)
    
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"\nR2 score on the test set: {round(score, 4)}")
    score = model.score(X_train, y_train)
    print(f"R2 score on the train set: {round(score, 4)}")

lin_reg()

