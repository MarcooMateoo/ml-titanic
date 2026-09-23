import pandas as pd

def load_dataset():
    train_set = pd.read_csv('../data/raw/train.csv')
    test_set = pd.read_csv('../data/raw/test.csv')

    return train_set, test_set


# shows missing entries from dataset
# use include_0 to treat value = 0 as null
def show_missing_entries(df, include_0 = False, include_0_cols=None):

    # initialize data frame as placeholder for include_0
    missing_0_df = pd.DataFrame()

    # include_0_cols must be a list
    if include_0 and not isinstance(include_0_cols, list): 
        raise ValueError("include_0_cols must be a list")

    # build a df for columns with 0 value
    if include_0:
        missing_0_df = (df == 0).sum()[include_0_cols].reset_index(name="Missing")
        missing_0_df = missing_0_df[missing_0_df['Missing'] > 0]

    # build a df for columns with missing values
    missing_df = df.isna().sum().reset_index(name='Missing')
    missing_df = missing_df[missing_df['Missing'] > 0]

    # if there is no missing or incorrect entries, don't return the final table
    if missing_df.shape[0] == 0 and missing_0_df.shape[0] == 0:
        return "No missing or incorrect entries"
        
    # concat columns with 0 value and NaN value
    missing_df = (pd.concat([missing_df, missing_0_df], axis = 0)
                  .rename(columns={'index': 'Column'})
                  .sort_values(by='Missing', ascending=False))
    
    # add percentage column
    missing_df['Percentage'] = round(100 * missing_df['Missing'] / df.shape[0], 2)
    
    return missing_df