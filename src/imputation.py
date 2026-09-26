from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

# Create a custom transformer for ImputeByGroupMean
class ImputeByGroup(BaseEstimator, TransformerMixin):
    def __init__(self, group_by_col, col_to_impute, impute_zero = False, strategy = 'mean'):
        self.group_by_col = group_by_col
        self.col_to_impute = col_to_impute
        self.impute_zero = impute_zero
        self.strategy = strategy

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        # strategy must only be median or mean
        if self.strategy not in ['median', 'mean']:
            raise ValueError("Strategy must be 'mean' or 'median'")

        # group_by_col must be a list
        if not isinstance(self.group_by_col, list):
            raise ValueError("group_by_col must be a list")

        # group_by_col must have at least 1 item
        if len(self.group_by_col) == 0:
            raise ValueError('group_by_col cannot be empty')

        # group_by_col must be inside the input DataFrame
        if not set(self.group_by_col).issubset(set(X.columns)):
            raise ValueError("group_by_col must be a subset of input columns")

        # column to impute must exist in the DataFrame
        if self.col_to_impute not in X.columns:
            raise ValueError("col_to_impute must be available in input columns")

        # col_to_impute must cotain atleast one non-null value
        if X[self.col_to_impute].isna().all():
            raise ValueError(f"{self.col_to_impute} contains only missing values")

        # impute_zero requires the col_to_impute to be numeric
        if not pd.api.types.is_numeric_dtype(X[self.col_to_impute]):
            raise ValueError(f"{self.col_to_impute} must be a numeric column")
        
        # record no. of missing values
        self.total_row_to_impute_ = (X[self.col_to_impute].isna() | 
                                    (self.impute_zero & (X[self.col_to_impute] == 0))).sum()

        if self.strategy == 'mean':
            self.group_stats_ = (X.groupby(self.group_by_col)[self.col_to_impute]
                                    .mean()
                                    .reset_index(name=self.col_to_impute))
        else:
            self.group_stats_ = (X.groupby(self.group_by_col)[self.col_to_impute]
                                    .median()
                                    .reset_index(name=self.col_to_impute))

        self.global_stats_ = (X[self.col_to_impute].mean() 
                                    if self.strategy == 'mean' 
                                    else X[self.col_to_impute].median())

        # as per sklearn convention, get feature name out should return numpy list
        self.feature_names_in_ = X.columns.to_numpy()
        
        return self
        
    def transform(self, X):

        # check if fit variables are set (trailing _ will be checked)
        check_is_fitted(
            self,
            ['group_stats_','global_stats_']
        )

        missing_cols = set([self.col_to_impute] + self.group_by_col) - set(X.columns)

        if missing_cols: 
            raise ValueError(f"Missing columns during transform: {missing_cols}")

        # duplicate the input df to avoid overriding values
        X = X.copy()

        # merge with the mean/median dataframe create during fit()
        X = pd.merge(X, 
                     self.group_stats_, 
                     on = self.group_by_col, 
                     how = 'left',
                     suffixes=("", "_group_stat")
                    )

        # fall back impute if there are NaN in group_stats
        X[f"{self.col_to_impute}_group_stat"] = X[f"{self.col_to_impute}_group_stat"].fillna(
            self.global_stats_
        )

        # fill na for the original column
        X[self.col_to_impute] = X[self.col_to_impute].fillna(
            X[f"{self.col_to_impute}_group_stat"]
        )

        # fill mean/median if original value is 0
        if self.impute_zero:
            X[self.col_to_impute] = np.where(
                X[self.col_to_impute] == 0,
                X[f"{self.col_to_impute}_group_stat"], 
                X[self.col_to_impute]
            )

        # drop extra columns
        X = X.drop(columns=[f"{self.col_to_impute}_group_stat"])

        return X

    def get_feature_names_out(self, input_features=None):
        # as per sklearn convention, get feature name out should return numpy list
        return (np.asarray(input_features)
                if input_features is not None 
                else self.feature_names_in_)



# Cabin Imputer
class FillCabin(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        X['Is_solo'] = ((X['SibSp'] + X['Parch'] + 1) == 1)
        X['Deck'] = X['Cabin'].str[0]

        self.global_stats_ = X.groupby(['Pclass', 'Is_solo'])['Deck'].agg(lambda X: X.mode()[0]).to_frame(name="Common_Deck")

        # as per sklearn convention, get feature name out should return numpy list
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X):
        X = X.copy()
        
        X['Is_solo'] = ((X['SibSp'] + X['Parch'] + 1) == 1)
        X = X.merge(self.global_stats_, on=['Pclass', 'Is_solo'], how='left')
        
        # X['Cabin'] = X['Cabin'].where(X['Cabin'].notna(), X['Common_Deck'])
        
        # X = X.drop(['Is_solo', 'Common_Deck', 'Deck'], axis = 1, errors='ignore')
        # print(X.value_counts('Cabin'))
        # as per sklearn convention, get feature name out should return numpy list
        self.feature_names_in_ = X.columns.to_numpy()
        
        return X 

    def get_feature_names_out(self, input_features=None):
        # as per sklearn convention, get feature name out should return numpy list
        return (np.asarray(input_features)
                if input_features is not None 
                else self.feature_names_in_)

class FillNA(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # as per sklearn convention, get feature name out should return numpy list
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X):
        if not set(['Cabin', 'Embarked']).issubset(set(X.columns)):
            raise ValueError("Cabin and Embarked must be a subset of input columns")
        X = X.copy()

        X['Embarked'] = X['Embarked'].fillna('S')
        X['Cabin'] = X['Cabin'].fillna('Unknown')
        
        return X 

    def get_feature_names_out(self, input_features=None):
        # as per sklearn convention, get feature name out should return numpy list
        return (np.asarray(input_features)
                if input_features is not None 
                else self.feature_names_in_)