# A linear regression baseline for preprocessed data

import Data_preprocessing as dp

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

data_prep = dp.Preprocessor()

X_train, X_test, y_train, y_test = data_prep.prepare_data()

@data_prep.timing
def lin_reg():

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"\nR2 score on the test set: {round(score, 4)}")
    score = model.score(X_train, y_train)
    print(f"R2 score on the train set: {round(score, 4)}")

@data_prep.timing
def lin_reg_cv():
    
    model = LinearRegression()
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
    
    print(f'\nR2 score on cross-validation: {round(scores.mean(), 4)}')

# lin_reg()
# lin_reg_cv()

