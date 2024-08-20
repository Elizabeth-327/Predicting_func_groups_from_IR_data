"""Creates training, validation, and test sets."""

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd
import pickle
from get_IRdata import IR_data


def split_data():
    """Splits data into n-fold train, validation, test sets for hyperparameter optimization.
    Written by Guwon Jung. Modifications made by me: removed identifying InChIs."""
    X, y = IR_data()

    data_dictionary = {}
    
    # Split data into training+val and test sets while preserving the proportion of each label (in this case, 0 or 1) in both sets
    # Create 2 different training+val and test splits
    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
    for train_val_index, test_index in msss.split(X, y):
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]

    # Add test set to dictionary
    data_dictionary['X_test'] = X_test
    data_dictionary['y_test'] = y_test

    # Create four-fold split of training and validation sets while preserving the proportion of each label in both sets for cross validation
    mskf = MultilabelStratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    num = 1 #fold count
    for train_index, val_index in mskf.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        data_dictionary['X_train_' + str(num)] = X_train
        data_dictionary['y_train_' + str(num)] = y_train
        data_dictionary['X_val_' + str(num)] = X_val
        data_dictionary['y_val_' + str(num)] = y_val
        num += 1

    # Save the dictionary as a pickle file
    with open('processed_dataset.pickle', 'wb') as handle:
        pickle.dump(data_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data_dictionary
if __name__ == '__main__': #runs only when this file is run, not when it is imported as a module elsewhere
    print(split_data())
    
    
                            
