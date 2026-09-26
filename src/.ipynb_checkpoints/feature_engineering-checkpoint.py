# Here are the list of features that could be a condidate for producing good signal
# This is a living list while doing EDA

# 1. Family_size = SibSp + Parch + 1
# 2. Is_solo = True If Family_size = 1 
# 3. Fare_bucket by 50 intervals
# 4. Age_bucket by 10 intervals
# 5. Deck = Cabin[0]
# 6. Is_deck_missing = True if Deck = U
# 3. Interaction features
#  a. Class_1_female
#  b. Class_2_male
#  c. Class_2_female
#  d. Class_3_male
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