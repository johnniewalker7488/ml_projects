# R2 on test: 0.8819
import Data_preprocessing as dp

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

data_prep = dp.Preprocessor()

X_train, X_test, y_train, y_test = data_prep.prepare_data()

@data_prep.timing
def random_forest():
    
    # param_grid = {'n_estimators': [100, 500], 'max_depth': [20, 50]}
    rfr = RandomForestRegressor(n_estimators=300, max_features=85, min_samples_leaf=5, n_jobs=-1, random_state=42)
    
    print('Random Forest is fitting...\n')
    rfr.fit(X_train, y_train)
    score = rfr.score(X_train, y_train) 
    print('R2 on train: ', round(score, 4))
    score = rfr.score(X_test, y_test)
    print('R2 on test: ', round(score, 4))

    # grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=3, n_jobs=-1, verbose=1)
    # print('GridSearchCV is fitting...\n')
    # grid_search.fit(X_train, y_train)
    
    # print('Best hyperparameters: ', grid_search.best_params_, '\n')

    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    #     print(round(mean_score, 4), params)

@data_prep.timing
def grad_reg():

    gbr = GradientBoostingRegressor(n_estimators=1000, max_depth=3, random_state=42, max_features=85)

    print('GBRegressor is fitting...')
    gbr.fit(X_train, y_train)
    score = gbr.score(X_train, y_train)
    print('R2 on train: ', round(score, 4))
    score = gbr.score(X_test, y_test)
    print('R2 on test: ', round(score, 4))


    # param_grid = {'n_estimators': [500, 1000], 'max_depth': [3, 6], 'max_features': [85, 254]}
    # grid_search = GridSearchCV(gbr, param_grid, scoring='r2', cv=3, n_jobs=-1, verbose=1)
    # print('GridSearchCV is fitting...\n')
    # grid_search.fit(X_train, y_train)
    
    # print('Best hyperparameters: ', grid_search.best_params_, '\n')
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    #     print(round(mean_score, 4), params)

