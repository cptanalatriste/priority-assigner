"""
This module builds a predictor for the priority field for a bug report
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

CSV_FILE = "C:\Users\Carlos G. Gavidia\git\priority-assigner\sample_file.csv"


def get_issues_dataframe(filename):
    """
    Returns a dataframe of issues that: Are resolived, have a commit in Git, were not resolved by the reported and had
    a priority change not made by the reporter.

    Also, that does not have the distance in releases null.

    :return: Filtered dataframe
    """
    issue_dataframe = pd.read_csv(filename)
    print "Loaded ", len(
        issue_dataframe.index), " resolved issues with Git Commits, solved by a third-party from ", filename

    issue_dataframe = issue_dataframe.dropna(subset=['Priority Changer', 'GitHub Distance in Releases'])
    print len(issue_dataframe.index), " issues had a change in the reported priority and release information in Git."

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


def escale_features(issues_train, issues_test, numerical_features):
    """
    Standarizes numerical information.
    :param issues_train: Train issue information.
    :param issues_test: Test issue information
    :param numerical_features: Features to scale

    :return: Train and test sets standarized.
    """

    issues_train_std = issues_train.copy()
    issues_test_std = issues_test.copy()

    for feature in numerical_features:
        scaler = StandardScaler()
        issues_train_std[feature] = scaler.fit_transform(issues_train[feature])
        issues_test_std[feature] = scaler.transform(issues_test[feature])

    return issues_train_std, issues_test_std


def select_features_l1(issues_train_std, priority_train, issues_test_std, priority_test):
    """
    Applies a Logistic Regression using L1-regularization, in order to get a sparse solution.
    :param issues_train_std: Issues for training.
    :param priority_train: Priorities for training.
    :param issues_test_std: Issues for testing.
    :param priority_test: Priorities for testing
    :return: None
    """
    logistic_regression = LogisticRegression(penalty='l1', C=0.1)
    logistic_regression.fit(issues_train_std, priority_train)

    priority_predictions = logistic_regression.predict(issues_test_std)

    print 'Training accuracy ', logistic_regression.score(issues_train_std, priority_train)
    print 'Test accuracy ', logistic_regression.score(issues_test_std, priority_test)

    print 'Intercept: ', logistic_regression.intercept_
    print 'Coefficient: ', logistic_regression.coef_

    print 'Test Precission: ', precision_score(y_true=priority_test, y_pred=priority_predictions)
    print 'Test Recall: ', recall_score(y_true=priority_test, y_pred=priority_predictions)
    print 'Test F1: ', f1_score(y_true=priority_test, y_pred=priority_predictions)

    train_predictions = logistic_regression.predict(issues_train_std)
    print 'Train Precission: ', precision_score(y_true=priority_train, y_pred=train_predictions)
    print 'Train Recall: ', recall_score(y_true=priority_train, y_pred=train_predictions)
    print 'Train F1: ', f1_score(y_true=priority_train, y_pred=train_predictions)


if __name__ == "__main__":
    issue_dataframe = get_issues_dataframe(CSV_FILE)

    class_label = 'Encoded Priority'
    numerical_features = ['Commits', 'GitHub Distance in Releases', 'Total LOC', 'Git Resolution Time',
                          'Comments in JIRA']
    nominal_features = ['Git Repository']

    issue_dataframe = encode_class_label(issue_dataframe, class_label)
    issue_dataframe = issue_dataframe[numerical_features + nominal_features + [class_label]]
    issue_dataframe = encode_nominal_features(issue_dataframe, nominal_features)

    encoded_priorities = issue_dataframe[class_label]
    issue_dataframe = issue_dataframe.drop([class_label], axis=1)

    issues_train, issues_test, priority_train, priority_test = train_test_split(issue_dataframe,
                                                                                encoded_priorities,
                                                                                test_size=0.3, random_state=0)

    print len(issues_train.index), " issues on the train set."

    issues_train_std, issues_test_std = escale_features(issues_train, issues_test, numerical_features)

    select_features_l1(issues_train_std, priority_train, issues_test_std, priority_test)
