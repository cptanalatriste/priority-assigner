"""
This modules deals with parameter tunning for the Priority Assigner
"""

import pandas as pd

from sklearn.cross_validation import train_test_split

import assigner

if __name__ == "__main__":
    original_dataframe = pd.read_csv(assigner.CSV_FILE)
    filtered_dataframe = assigner.filter_issues_dataframe(original_dataframe)

    issues, priorities = assigner.prepare_for_training(filtered_dataframe, assigner.CLASS_LABEL,
                                                       assigner.NUMERICAL_FEATURES, assigner.NOMINAL_FEATURES)

    issues_train, issues_test, priority_train, priority_test = train_test_split(issues, priorities)
    issues_train_std, issues_test_std = assigner.escale_numerical_features(assigner.NUMERICAL_FEATURES, issues_train,
                                                                           issues_test)
