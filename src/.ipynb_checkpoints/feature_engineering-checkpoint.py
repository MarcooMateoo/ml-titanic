# Here are the list of features that could be a condidate for producing good signal
# This is a living list while doing EDA

# 1. Family_size = SibSp + Parch + 1
# 2. Is_solo = True If Family_size = 1 
# 3. Fare_bucket by 50 intervals
# 4. Age_bucket by 10 intervals
# 5. Deck = Cabin[0]
# 6. Is_deck_missing = True if Deck = U
# 7. Title = common title Mr, Miss, Mrs, Master, Rare otherwise
# 8. Is_ticket_num
# 9. Ticket_first_char


# Interaction features
#  a. [Hurt] Class_1_female 
#  b. [Hurt] Class_2_male 
#  c. [Hurt] Class_2_female 
#  d. [Gain] Class_3_male
#  e. Embarked_S_male
#  f. Embarked_S_female
#  g. Embarked_C_c3
#  h. Embarked_S_Age_21-30
#  i. Class_3_Age_21-30
#  j. Male_Age_21-30
#  k. Female_Age_21-30
#  l. Female_Age_31-40
#  m. Class_3_Deck_U
#  n. Male_Deck_U
#  o. Embarked_S_Deck_U
#  p. Class_3_Solo
#  q. Male_Solo
#  r. Embarked_S_Solo
#  s. [Gain] Class_1_master



# ======= Start =======

# For custom transformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Family_size
class FeatFamilySize(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Family_size')
        return self

    def transform(self, X):
        X = X.copy()
        X['Family_size'] = X['SibSp'] + X['Parch'] + 1
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# Is_solo
class FeatIsSolo(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Is_solo')
        return self

    def transform(self, X):
        X = X.copy()
        X['Is_solo'] = ((X['SibSp'] + X['Parch'] + 1) == 1)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# Fare_bucket by 50 intervals
class FeatFareBucket(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Fare_bucket')
        return self

    def transform(self, X):
        X = X.copy()
        X['Fare_bucket'] = pd.cut(
                                X['Fare'],
                                bins=[0, 50, 100, 150, 200, 250, float('inf')],
                                labels=[
                                    '0-50',
                                    '51-100',
                                    '101-150',
                                    '151-200',
                                    '201-250',
                                    '251+'
                                    ],
                                include_lowest=True
                            )
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# Age_bucket by 10 intervals
class FeatAgeBucket(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Age_bucket')
        return self

    def transform(self, X):
        X = X.copy()
        X['Age_bucket'] = pd.cut(
            X['Age'],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, float('inf')], 
            labels=[
                '0-10', '11-20', '21-30', '31-40', '41-50', 
                '51-60', '61-70', '71-above'
            ],
            include_lowest=True
        )
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# Deck = Cabin[0]
class FeatDeck(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Deck')
        return self

    def transform(self, X):
        X = X.copy()
        X['Deck'] = X['Cabin'].str[0]
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# Is_deck_missing = True if Deck = U
class FeatDeckMissing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Is_deck_missing')
        return self

    def transform(self, X):
        X = X.copy()
        
        X['Deck_missing'] = (X['Cabin'].isna()) | (X['Deck'] == 'U')
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# Title = common title Mr, Miss, Mrs, Master, Rare otherwise
class FeatTitle(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Title')
        return self

    def transform(self, X):
        X = X.copy()
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        X['Title'] = X["Name"].str.extract(
              r',\s*(.*?)\.',
              expand=False
        )

        X['Title'] = X['Title'].where(X['Title'].isin(common_titles), 'Rare')
        
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# Is_ticket_num
class FeatIsTicketNum(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Is_ticket_num')
        return self

    def transform(self, X):
        X = X.copy()
        X['Is_ticket_num'] = (X['Ticket'].str.match(r'[A-Za-z]') == False)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# Ticket_char_0
class FeatTicketFirstChar(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Ticket_str_0')
        return self

    def transform(self, X):
        X = X.copy()
        X['Ticket_char_0'] = X['Ticket'].str[0]
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# Is_adult
class FeatIsAdult(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, 'Is_adult')
        return self

    def transform(self, X):
        X = X.copy()
        X['Is_adult'] = (X['Age'] > 16).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop
        
    def fit(self, X, y=None):
        self.feature_names_out_ = np.array([c for c in X.columns if c not in self.cols_to_drop])
        return self
        
    def transform(self, X):
        X = X.copy()
        X = X.drop(labels = self.cols_to_drop, errors='ignore', axis = 1)
        
        return X
        
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_)


# FILL CABIN IMPUTATION
#  a. [Hurt] Class_1_female 
#  b. [Hurt] Class_2_male 
#  c. [Hurt] Class_2_female 
#  d. [Gain] Class_3_male
#  e. [Hurt] Embarked_S_male
#  f. [Hurt] Embarked_S_female
#  g. [Hurt] Embarked_C_c3
#  h. [Hurt] Embarked_S_Age_21-30
#  i. [Gain] Class_3_Age_21-30
#  j. [Hurt] Male_Age_21-30
#  k. [Hurt] Female_Age_21-30
#  l. [Gain] Female_Age_31-40
#  m. Class_3_Deck_U
#  n. Male_Deck_U
#  o. Embarked_S_Deck_U
#  p. [Hurt] Class_3_Solo
#  q. [Hurt] Male_Solo
#  r. [Hurt] Embarked_S_Solo
#  s. [Gain] Class_1_master


# NO FILL CABIN IMPUTATION
#  a. [Gain] Class_1_female 
#  b. [Hurt] Class_2_male 
#  c. [Hurt] Class_2_female 
#  d. [Gain] Class_3_male
#  e. [Hurt] Embarked_S_male
#  f. [Hurt] Embarked_S_female
#  g. [Hurt] Embarked_C_c3
#  h. [Hurt] Embarked_S_Age_21-30
#  i. [Hurt] Class_3_Age_21-30
#  j. [Gain] Male_Age_21-30
#  k. [Gain] Female_Age_21-30
#  l. [Gain] Female_Age_31-40
#  m. [Hurt] Class_3_Deck_U
#  n. [Hurt] Male_Deck_U
#  o. [Hurt] Embarked_S_Deck_U
#  p. [Gain] Class_3_Solo
#  q. [Hurt] Male_Solo
#  r. [Hurt] Embarked_S_Solo
#  s. [Hurt] Class_1_master

class FeatInteraction(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
         # as per sklearn convention, get feature name out should return numpy list
        curr_col = X.columns.to_numpy()
        self.feature_names_in_ = np.append(curr_col, ['Class_1_master', 'Class_3_male'])
        return self

    def transform(self, X):
        X = X.copy()
        X['Class_3_male'] = ((X['Sex'] == 'male') & (X['Pclass'] == 3)) 
        X['Class_1_female'] = ((X['Sex'] == 'female') & (X['Pclass'] == 1))
        X['Male_Age_21-30'] = ((X['Sex'] == 'male') & (X['Age'] >= 21) & (X['Age'] <= 30))
        X['Female_Age_21-30'] = ((X['Sex'] == 'female') & (X['Age'] >= 21) & (X['Age'] <= 30))
        X['Female_Age_31-40'] = ((X['Sex'] == 'female') & (X['Age'] >= 31) & (X['Age'] <= 40))
        X['Class_3_Solo'] = ((X['Pclass'] == 3) & (X['Is_solo'] == True)) 
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_