import pandas as pd
import numpy as np
from ml import fetch_housing_data, load_housing_data
from matplotlib import pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def validate_and_report(*args, name='', **kwargs):
    scores = cross_val_score(*args, **kwargs)
    rmse_error = np.sqrt(-1*scores)
    rmse_mean = np.mean(rmse_error)
    rmse_std = np.std(rmse_error)
    print(f'{name} Model RMSE: {rmse_mean:.0f}, std: {rmse_std:.0f}')

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Adds more meaningful attributes to our data. In the future, should work on data indices rather than
    names, as a pipeline would work only with numpy arrays internally.
    """
    rooms_index = 3
    bedrooms_index = 4
    population_index = 5
    households_index = 6

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, self.rooms_index]/X[:, self.households_index]
        population_per_household = X[:, self.population_index]/X[:, self.households_index]
        if self.add_bedrooms_per_room == True:
            bedrooms_per_room = X[:, self.bedrooms_index]/X[:, self.rooms_index]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

