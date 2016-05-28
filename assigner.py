"""
This module builds a predictor for the priority field for a bug report
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

CSV_FILE = "C:\Users\Carlos G. Gavidia\git\priority-assigner\sample_file.csv"


def get_issues_dataframe(filename):
    """
    Returns a dataframe of issues that: Are resolived, have a commit in Git, were not resolved by the reported and had
    a priority change not made by the reporter.
    :return: Filtered dataframe
    """
    issue_dataframe = pd.read_csv(filename)
    print "Loaded ", len(
        issue_dataframe.index), " resolved issues with Git Commits, solved by a third-party from ", filename

    issue_dataframe = issue_dataframe.dropna(subset=['Priority Changer'])
    print len(issue_dataframe.index), " issues had a change in the reported priority."

    issue_dataframe = issue_dataframe[issue_dataframe['Priority Changer'] != issue_dataframe['Reported By']]
    print len(issue_dataframe.index), " issues had a priority corrected by a third-party."
    return issue_dataframe


def encode_class_label(issue_dataframe, class_label):
    """
    Replaces the priority as String for a numerical value.
    :param issue_dataframe: Original dataframe
    :param class_label: Column containing Priority Information
    :return: New dataframe
    """
    priority_mapping = {'Blocker': 1,
                        'Critical': 2,
                        'Major': 3,
                        'Minor': 4,
                        'Trivial': 5}

    issue_dataframe[class_label] = issue_dataframe['Priority'].map(priority_mapping)
    return issue_dataframe


def encode_nominal_features(issue_dataframe, nominal_features):
    """
    Performs a one-hot encoding on the nominal features of the dataset.
    :param issue_dataframe: Original Dataframe
    :param nominal_features: List of nominal features.
    :return: New dataframe.
    """
    dummy_dataframe = pd.get_dummies(issue_dataframe[nominal_features])

    temp_dataframe = pd.concat([issue_dataframe, dummy_dataframe], axis=1)
    temp_dataframe = temp_dataframe.drop(nominal_features, axis=1)

    return temp_dataframe


if __name__ == "__main__":
    issue_dataframe = get_issues_dataframe(CSV_FILE)

    class_label = 'Encoded Priority'
    numerical_features = ['Commits', 'GitHub Distance in Releases', 'Total LOC', 'Git Resolution Time',
                          'Comments in JIRA']
    nominal_features = ['Git Repository']

    issue_dataframe = encode_class_label(issue_dataframe, class_label)
    issue_dataframe = issue_dataframe[numerical_features + nominal_features + [class_label]]
    issue_dataframe = encode_nominal_features(issue_dataframe, nominal_features)

    print issue_dataframe.head()
