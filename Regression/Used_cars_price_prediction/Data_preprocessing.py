import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

from functools import wraps
import time

filename = 'Git/ml_projects/Regression/Used_cars_price_prediction/true_car_listings.csv'
save_path = 'Git/ml_projects/Regression/Used_cars_price_prediction/'

class Preprocessor:
    """Used to preprocess data for further training"""
    ################################################################################################################
    def import_data(self, filename):
        """ Imports given csv file as a pandas dataframe """
        df_raw = pd.read_csv(filename)
        
        return df_raw
    #################################################################################################################
    def clean_data(self, dataframe):

        """ Performs data cleaning for a given pandas dataframe:
                - checks for NaNs
                - removes outliers
                - encodes 'City' column using us_cities dataframe
                - prepares categorical features for encoding
                
            Args: pandas dataframe from true_car_listings.csv

            Returns: pandas dataframe with cleaned data
        """

        # Checking for missing values
        print('Missing values found: ', dataframe.isnull().values.any())

        # Removing outliers - setting up a constraint of 3 standard deviations for price and mileage
        std_dev = 3
        df_clean = dataframe[(np.abs(stats.zscore(dataframe[['Price', 'Mileage']])) < float(std_dev)).all(axis=1)]

        # Use 'US City Populations.csv' to encode the 'City' column
        
        us_cities = pd.read_excel('Git/ml_projects/Regression/Used_cars_price_prediction/US City Populations.xlsx')
        us_cities.drop('State', axis=1, inplace=True)
        us_cities.sort_values('Population', ascending=False)
        us_cities.drop_duplicates(subset='City', keep='first', inplace=True)
        df_clean = df_clean.merge(us_cities, on='City', how='left')
        
        # Impute missing values to 'Population' column
        df_clean.Population.fillna(df_clean['Population'].median(), inplace=True)

        # Encode the city column with 1 for city population more than 50000 and 0 for less 50000
        # and remove the unnecessary columns
        df_clean['City'] = np.where(df_clean['Population'] > 50000, 1, 0)
        
        df_clean.drop(['Vin', 'Population'], axis=1, inplace=True)

        # Preprocess the categorical columns
        df_clean['State'] = df_clean['State'].str.lower()
        df_clean['Make'] = df_clean['Make'].str.lower()
        df_clean['Model'] = df_clean['Model'].str.lower()
        df_clean['Model'] = df_clean['Model'].str.replace(' ', '')
        df_clean['Model'] = df_clean['Model'].str.replace('-', '')

        # Dropping rare models
        cars_drop = df_clean['Model'].value_counts()[171:]
        df_clean.drop(df_clean[df_clean.Model.isin(cars_drop.index)].index, inplace=True)

        print('\nCleaned data shape: ', df_clean.shape)
        df_clean.to_csv(save_path + 'Clean_data.csv', encoding='utf-8', index=False)

        return df_clean
    #################################################################################################################
    def visualize_prep(self, raw_data, clean_data):
        """ Plots histograms of numeric values before and after outliers removal """

        # Histograms of numeric features before outliers removal
        plt.figure(figsize=(20,8))
        plt.subplot(1, 3, 1)
        plt.hist(raw_data['Price'], bins=100)
        plt.title('Price')

        plt.subplot(1, 3, 2)
        plt.hist(raw_data['Mileage'], bins=100)
        plt.title('Mileage')

        plt.subplot(1, 3, 3)
        plt.hist(raw_data['Year'], bins=100)
        plt.title('Year')

        plt.savefig(save_path + 'Numeric_before_preprocessing.png')
        print('Histograms for numeric values before preprocessing saved to current directory')

        # Histograms of numeric features before outliers removal
        plt.figure(figsize=(20,8))
        plt.subplot(1, 3, 1)
        plt.hist(clean_data['Price'], bins=100)
        plt.title('Price')

        plt.subplot(1, 3, 2)
        plt.hist(clean_data['Mileage'], bins=100)
        plt.title('Mileage')

        plt.subplot(1, 3, 3)
        plt.hist(clean_data['Year'], bins=100)
        plt.title('Year')

        plt.savefig(save_path + 'Numeric_after_preprocessing.png')
        print('Histograms for numeric values after preprocessing saved to current directory')

        # Take a particular car to show linearity
        honda = clean_data[(clean_data.Make == 'honda') & (clean_data.Model == 'accord')]

        plt.figure(figsize=(6, 4))
        sns.scatterplot(x='Mileage', y='Price', data=honda, marker='x')
        plt.savefig(save_path + 'Mileage_vs_price.png')
        print('Mileage vs price plot saved to current directory')

        plt.figure(figsize=(6, 4))
        sns.scatterplot(x='Year', y='Price', data=honda, marker='x')
        plt.savefig(save_path + 'Year_vs_price.png')
        print('Year vs price plot saved to current directory')        
    #################################################################################################################
    def split_data(self, dataframe, test_size, random_state=42):
        """Splits the data into train and test sets"""

        X = dataframe.drop('Price', axis=1)
        y = dataframe['Price']

        # X_sample = X.sample(frac=0.15, random_state=42)
        # y_sample = y.sample(frac=0.15, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print('\nData split into train and test sets')
        
        return X_train, X_test, y_train, y_test
    #################################################################################################################
    def scale_numeric(self, X_train, X_test):
        """Scales numeric data"""

        scaler = MinMaxScaler()
        X_train.loc[:, ['Year', 'Mileage']] = scaler.fit_transform(X_train.loc[:, ['Year', 'Mileage']])
        X_test.loc[:, ['Year', 'Mileage']] = scaler.transform(X_test.loc[:, ['Year', 'Mileage']])

        print('\nNumeric features scaled')
        
        return X_train, X_test
    #################################################################################################################
    def one_hot(self, X_train, X_test):
        """Splits the data into train and test with one-hot encoding of categorical features"""

        
        column_trans = make_column_transformer((OneHotEncoder(), ['State', 'Make', 'Model']),
                                                remainder='passthrough')
        
        X_train = column_trans.fit_transform(X_train)
        print('X_train encoded shape: ', X_train.shape)
        X_test = column_trans.fit_transform(X_test)
        print('X_test encoded shape: ', X_test.shape)

        return X_train, X_test
    #################################################################################################################
    def timing(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            run_time = time.time() - start_time
            print(f'{func.__name__} ran in {round(run_time / 60, 2)} minutes')
            return result

        return wrapper

    #################################################################################################################

    def prepare_data(self):
        filename = 'Git/ml_projects/Regression/Used_cars_price_prediction/true_car_listings.csv'
        df_raw = self.import_data(filename)
        df_clean = self.clean_data(df_raw)
        X_train, X_test, y_train, y_test = self.split_data(df_clean, test_size=0.2, random_state=42)
        X_train, X_test = self.scale_numeric(X_train, X_test)
        X_train, X_test = self.one_hot(X_train, X_test)

        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_prep = Preprocessor()
    X_train, X_test, y_train, y_test = data_prep.prepare_data()
    
    print('\nData has been preprocessed for training\n')
    # data_prep.visualize_prep(df_raw, df_clean)