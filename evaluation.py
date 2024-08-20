"""Demonstrates evaluation of the model."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os

import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, average_precision_score
from optimal_thresholding import optimal_threshold


def model_predict(X_test, y_train_val, loaded_model, optimal_thresh, optimal):
    """Predicts probabilities for all classes for a given model. Written by Guwon Jung."""
    # Prediction probabilities.
    y_probabilities = loaded_model.predict(X_test)

    # Classify probabilities.
    y_pred = []
    # Apply a threshold of 0.5.
    if optimal == 0:
        for prob in y_probabilities:
            y_pred.append([1 if k >= 0.5 else 0 for k in prob])
    # Apply optimal thresholds.
    elif optimal == 1:
        for prob in y_probabilities:
            single = []
            for i in range(y_train_val.shape[1]):
                if prob[i] >= optimal_thresh[i]:
                    single.append(1)
                else:
                    single.append(0)
            y_pred.append(single)

    return y_pred


def f_score(y_test, y_pred, y_train_val, label_names):
    """Calculates the F1-score, precision, and recall. Written by Guwon Jung."""
    fs, pr, re = [], [], []
    for i in range(len(label_names)):
        temp_test = [sample[i] for sample in y_test]
        temp_pred = [sample[i] for sample in y_pred]

        fs.append(f1_score(temp_test, temp_pred))
        pr.append(precision_score(temp_test, temp_pred))
        re.append(recall_score(temp_test, temp_pred))

    data = pd.DataFrame(y_train_val)
    count = [int(sum(data[i])) for i in range(len(label_names))]

    names = [label for _, label in sorted(zip(fs, label_names))]
    re_fs = sorted(fs)
    re_pr, re_rc, re_count = [], [], []

    for name in names:
        idx = label_names.index(name)
        re_pr.append(pr[idx])
        re_rc.append(re[idx])
        re_count.append(count[idx])

    result = pd.DataFrame(
        {'F-score': re_fs,
         'Precision': re_pr,
         'Recall': re_rc,
         'Frequency': re_count
        }, index=names)
    result.index.name = 'FGs'
    print(result)
    result.to_csv('f_score_results.csv')

    return re_fs, re_pr, re_rc, re_count, names

 
if __name__ == '__main__':
    # Get func groups
    all_func_groups = []
    with open('all_unique_groups.txt', 'r') as handle:
        for line in handle:
            func_group = line.strip()
            all_func_groups.append(func_group)

    # Read data.
    with open('processed_dataset.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)

    # Load model.
    loaded_model = load_model('./models/0_model_original.h5')

    # Combined training and validation sets.
    X_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['X_train_1']), pd.DataFrame(dict_data['X_val_1'])])).astype('float32')
    y_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['y_train_1']), pd.DataFrame(dict_data['y_val_1'])])).astype('float32')
    # Define test set.
    X_test = np.asarray(pd.DataFrame(dict_data['X_test'])).astype('float32')
    y_test = np.asarray(pd.DataFrame(dict_data['y_test'])).astype('float32')
    # Shape input data into three dimensions.
    X_test = X_test.reshape(X_test.shape[0], 600, 1)

    # Calculate optimal thresholds for each functional group.
    optimal_thresh = optimal_threshold(X_train_val, y_train_val)

    # Make predictions.
    y_pred = model_predict(X_test, y_train_val, loaded_model, optimal_thresh, 0)

    # Display evaluation.
    # F-score, precision, recall, and sample frequency.
    print('Calculating F-score, precision, recall, and sample frequency of functional groups (FGs) in Figure 2.\n')
    f_score(y_test, y_pred, y_train_val, all_func_groups)

  