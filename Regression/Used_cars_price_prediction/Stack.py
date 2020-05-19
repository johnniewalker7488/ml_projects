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
from sklearn.metrics import mean_squared_error

data_prep = dp.Preprocessor()

X_train, X_test, y_train, y_test = data_prep.prepare_data()

def lin_reg(X_train, X_test, y_train, y_test):

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f'\nRMSE on test: {round(rmse, 2)}')

    score = model.score(X_train, y_train)
    print(f"\nR2 score on the train set: {round(score, 4)}")
    score = model.score(X_test, y_test)
    print(f"\nR2 score on the test set: {round(score, 4)}")
    

@data_prep.timing
def random_forest():
    
    
    rfr = RandomForestRegressor(n_estimators=300, max_features=85, min_samples_leaf=5, n_jobs=-1, random_state=42)
    
    print('Random Forest is fitting...\n')
    rfr.fit(X_train, y_train)
    score = rfr.score(X_train, y_train) 
    print('R2 on train: ', round(score, 4))
    score = rfr.score(X_test, y_test)
    print('R2 on test: ', round(score, 4))

    y_pred_train_rfr = rfr.predict(X_train)
    y_pred_test_rfr = rfr.predict(X_test)

    return y_pred_train_rfr, y_pred_test_rfr

    # param_grid = {'n_estimators': [100, 500], 'max_depth': [20, 50]}
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

    print('\nGBRegressor is fitting...\n')
    gbr.fit(X_train, y_train)
    score = gbr.score(X_train, y_train)
    print('R2 on train: ', round(score, 4))
    score = gbr.score(X_test, y_test)
    print('R2 on test: ', round(score, 4))

    y_pred_train_gbr = gbr.predict(X_train)
    y_pred_test_gbr = gbr.predict(X_test)

    return y_pred_train_gbr, y_pred_test_gbr

    # param_grid = {'n_estimators': [500, 1000], 'max_depth': [3, 6], 'max_features': [85, 254]}
    # grid_search = GridSearchCV(gbr, param_grid, scoring='r2', cv=3, n_jobs=-1, verbose=1)
    # print('GridSearchCV is fitting...\n')
    # grid_search.fit(X_train, y_train)
    
    # print('Best hyperparameters: ', grid_search.best_params_, '\n')
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    #     print(round(mean_score, 4), params)


def stacking():
    
    y_pred_train_rfr, y_pred_test_rfr = random_forest()
    y_pred_train_gbr, y_pred_test_gbr = grad_reg()

    pred_df_train = pd.DataFrame({'pred_rfr': y_pred_train_rfr, 'pred_gbr': y_pred_train_gbr})
    pred_df_test = pd.DataFrame({'pred_rfr': y_pred_test_rfr, 'pred_gbr': y_pred_test_gbr})

    pred_df_train.reset_index(drop=True, inplace=True)
    pred_df_test.reset_index(drop=True, inplace=True)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    X_train_2 = pd.concat([X_train, pred_df_train], axis=1)
    X_test_2 = pd.concat([X_test, pred_df_test], axis=1)

    # save_path = 'Git/ml_projects/Regression/Used_cars_price_prediction/'
    # X_train_2.to_csv(save_path + 'Stacked_train.csv', encoding='utf-8', index=False)
    # X_test_2.to_csv(save_path + 'Stacked_test.csv', encoding='utf-8', index=False)

    lin_reg(X_train_2, X_test_2, y_train, y_test)



stacking()


