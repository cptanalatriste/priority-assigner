"""
This module builds a predictor for the priority field for a bug report
"""

import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np

import fselect

CSV_FILE = "C:\Users\Carlos G. Gavidia\git\github-data-miner\Release_Counter_.csv"


def filter_issues_dataframe(original_dataframe):
    """
    Returns a dataframe of issues that: Are resolved, have a commit in Git, were not resolved by the reported and had
    a priority change not made by the reporter.

    Also, that does not have the distance in releases null.

    :return: Filtered dataframe
    """

    print "Total columns in dataframe: ", len(original_dataframe.columns)

    issue_dataframe = original_dataframe.dropna(subset=['Priority Changer', 'GitHub Distance in Releases'])
    print len(issue_dataframe.index), " issues had a change in the reported priority and release information in Git."

    issue_dataframe = issue_dataframe[issue_dataframe['Priority Changer'] != issue_dataframe['Reported By']]
    print len(issue_dataframe.index), " issues had a priority corrected by a third-party."

    return issue_dataframe


def encode_class_label(issue_dataframe, class_label):
    """
    Replaces the priority as String for a numerical value. It also adds the Severe column (Boolean, true if it is a Blocker or Critical issue.)
    :param issue_dataframe: Original dataframe
    :param class_label: Column containing encoded Priority Information
    :return: New dataframe
    """
    priority_mapping = {'Blocker': 1,
                        'Critical': 2,
                        'Major': 3,
                        'Minor': 4,
                        'Trivial': 5}

    original_label = 'Priority'
    issue_dataframe[class_label] = issue_dataframe[original_label].map(priority_mapping)
    issue_dataframe['Severe'] = issue_dataframe[class_label] <= 2

    simplified_mapping = {'Blocker': 1,
                          'Critical': 1,
                          'Major': 2,
                          'Minor': 3,
                          'Trivial': 3}
    issue_dataframe['Simplified ' + class_label] = issue_dataframe[original_label].map(simplified_mapping)

    return issue_dataframe


def encode_nominal_features(issue_dataframe, nominal_features):
    """
    Performs a one-hot encoding on the nominal features of the dataset.
    :param issue_dataframe: Original Dataframe
    :param nominal_features: List of nominal features.
    :return: New dataframe.
    """

    if nominal_features:
        dummy_dataframe = pd.get_dummies(issue_dataframe[nominal_features])
        temp_dataframe = pd.concat([issue_dataframe, dummy_dataframe], axis=1)
        temp_dataframe = temp_dataframe.drop(nominal_features, axis=1)

        return temp_dataframe

    else:
        return issue_dataframe


def escale_features(numerical_features, issues_train, issues_test):
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

        print "feature ", feature
        issues_test_std[feature] = scaler.transform(issues_test[feature])

    return issues_train_std, issues_test_std


def evaluate_performance(prefix, classifier, issues_train, priority_train, issues_test_std, priority_test):
    """
    Calculates performance metrics for a classifier.
    :param prefix: A prefix, for identifying the classifier.
    :param classifier: The classifier.
    :param issues_train: Train features.
    :param priority_train: Train class.
    :param issues_test_std: Test features.
    :param priority_test: Test class.
    :return: None.
    """
    print prefix, ': Training accuracy ', classifier.score(issues_train, priority_train)
    train_predictions = classifier.predict(issues_train)
    print prefix, " :TRAIN DATA SET"
    print classification_report(y_true=priority_train, y_pred=train_predictions)

    print prefix, ': Test accuracy ', classifier.score(issues_test_std, priority_test)
    test_predictions = classifier.predict(issues_test_std)
    print prefix, " :TEST DATA SET"
    print classification_report(y_true=priority_test, y_pred=test_predictions)


def select_features_l1(issues_train_std, priority_train, issues_test_std, priority_test):
    """
    Applies a Logistic Regression using L1-regularization, in order to get a sparse solution.
    :param issues_train_std: Issues for training.
    :param priority_train: Priorities for training.
    :param issues_test_std: Issues for testing.
    :param priority_test: Priorities for testing
    :return: The Logistic Regression classifier.
    """
    logistic_regression = LogisticRegression(penalty='l1', C=0.1)
    logistic_regression.fit(issues_train_std, priority_train)

    # print 'Intercept: ', logistic_regression.intercept_
    # print 'Coefficient: ', logistic_regression.coef_

    evaluate_performance("LOGIT-L1", logistic_regression, issues_train_std, priority_train, issues_test_std,
                         priority_test)

    return logistic_regression


def sequential_feature_selection(issues_train_std, priority_train, issues_test_std, priority_test):
    """
    Applies a sequential feature selection algorithm and evaluates its performance using a k-neighbors classifier (5 neighbors).
    :param issues_train_std: Train features.
    :param priority_train: Train classes.
    :param issues_test_std: Test features.
    :param priority_test: Test classes.
    :return: None.
    """

    # TODO: What is the optimal number of neighbors?
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    feature_selector = fselect.SBS(knn_classifier, k_features=1)
    feature_selector.fit(issues_train_std.values, priority_train.values)

    optimal_subset = None
    for subset in feature_selector.subsets_:

        if len(subset) == 5:
            print "Number of features in subset: ", len(subset)
            print issues_train_std.columns[list(subset)]
            optimal_subset = list(subset)

    num_features = [len(k) for k in feature_selector.subsets_]
    plt.plot(num_features, feature_selector.scores_, marker='o')
    plt.ylim([0.4, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()

    # Uncomment to display plot.
    # plt.show()

    knn_classifier.fit(issues_train_std, priority_train)
    evaluate_performance("KNN-5NEIGH", knn_classifier, issues_train_std, priority_train, issues_test_std, priority_test)

    new_train = issues_train_std.iloc[:, optimal_subset]
    new_test = issues_test_std.iloc[:, optimal_subset]

    knn_classifier.fit(new_train, priority_train)
    evaluate_performance("KNN-OPT", knn_classifier, new_train, priority_train, new_test, priority_test)

    return knn_classifier


def feature_importance_with_forest(issues_train, priority_train, issues_test, priority_test):
    """
    Assess feature importance using a Random Forest.
    :param issues_train: Train features.
    :param priority_train: Train classes.
    :param issues_test: Test features.
    :param priority_test: Test classes.
    :return: None
    """
    print "Building Random Forest Classifier ..."
    rforest_classifier = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    rforest_classifier.fit(issues_train, priority_train)

    importances = rforest_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    for column_index in range(len(issues_train.columns)):
        print column_index + 1, ") ", issues_train.columns[column_index], " ", importances[indices[column_index]]

    plt.title('Feature importance')
    plt.bar(range(len(issues_train.columns)), importances[indices], color='lightblue', align='center')
    plt.xticks(range(len(issues_train.columns)), issues_train.columns, rotation=90)
    plt.xlim([-1, len(issues_train.columns)])
    plt.tight_layout()
    # plt.show()

    evaluate_performance("FOREST", rforest_classifier, issues_train, priority_train, issues_test, priority_test)

    print "Selecting important features ..."
    select = SelectFromModel(rforest_classifier, threshold=0.05, prefit=True)

    train_selected = select.transform(issues_train)
    test_selected = select.transform(issues_test)

    rforest_classifier.fit(train_selected, priority_train)
    evaluate_performance("FOREST-IMPORTANT", rforest_classifier, train_selected, priority_train, test_selected,
                         priority_test)


def prepare_for_training(issues_dataframe, class_label, numerical_features, nominal_features):
    """
    Filters only the relevant features, encodes de class label and the encodes nominal features.
    :param issues_dataframe: Original dataframe.
    :param class_label: Class label.
    :param numerical_features: Numerical features.
    :param nominal_features: Nominal features.
    :return: Dataframe with features, series with labels.
    """
    issue_dataframe = encode_class_label(issues_dataframe, class_label)

    class_label = 'Severe'
    # class_label = 'Simplified ' + class_label

    issue_dataframe = issue_dataframe[numerical_features + nominal_features + [class_label]]
    issue_dataframe = encode_nominal_features(issue_dataframe, nominal_features)

    encoded_priorities = issue_dataframe[class_label]
    issue_dataframe = issue_dataframe.drop([class_label], axis=1)

    print "Number of features: ", len(issue_dataframe.columns)

    return issue_dataframe, encoded_priorities


def train_and_predict(classifier, original_dataframe, training_dataframe, training_labels, class_label,
                      numerical_features,
                      nominal_features):
    """
    Taking an issues dataframe, it performs predictions based on a classifier.It writes the resulting dataframe to a file.
    :param classifier: Classifier to use.
    :param original_dataframe: Dataframe to predict.
    :param training_dataframe: Dataframe containing training instances.
    :param training_labels: Labels for the training dataset.
    :param class_label: Class label.
    :param numerical_features: Numerical Features.
    :param nominal_features: Nominal features.
    :return:
    """

    print "Missing data analysis ..."
    print original_dataframe.isnull().sum()

    # Temporarly, we are dropping NA values
    temp_dataframe = original_dataframe.dropna(subset=['GitHub Distance in Releases', 'Git Resolution Time'])

    # The following repositories were not in the training dataset
    repository_label = 'Git Repository'
    temp_dataframe = temp_dataframe[~temp_dataframe[repository_label].isin(['kylin', 'helix', 'mesos'])]

    print "Starting prediction process..."
    issues_dataframe, encoded_priorities = prepare_for_training(temp_dataframe, class_label, numerical_features,
                                                                nominal_features)

    training_std, original_std = escale_features(numerical_features, training_dataframe, issues_dataframe)

    classifier.fit(training_std, training_labels)
    print "Training score: ", classifier.score(training_std, training_labels)
    train_predictions = classifier.predict(training_std)
    print classification_report(y_true=training_labels, y_pred=train_predictions)

    print "Predicting ", len(original_std.index), " issue's priority"
    test_predictions = classifier.predict(original_std)

    predicted_label = 'Predicted ' + class_label
    temp_dataframe[predicted_label] = test_predictions

    file_name = "Including_Prediction.csv"
    temp_dataframe.to_csv(file_name, index=False)

    print "File ready: ", file_name

    for repository in temp_dataframe[repository_label].unique():
        issues_for_repo = temp_dataframe[temp_dataframe[repository_label] == repository]
        severe_issues = issues_for_repo[issues_for_repo['Severe']]
        inflated_issues = severe_issues[~severe_issues[predicted_label]]

        print "repository: ", repository, " Reported Severe: ", len(severe_issues.index), " Severe inflated: ", len(
            inflated_issues.index), ' ratio: ', len(inflated_issues.index) / float(len(severe_issues.index))


if __name__ == "__main__":
    original_dataframe = pd.read_csv(CSV_FILE)

    print "Loaded ", len(
        original_dataframe.index), " resolved issues with Git Commits, solved by a third-party from ", CSV_FILE

    issues_dataframe = filter_issues_dataframe(original_dataframe)

    class_label = 'Encoded Priority'
    numerical_features = ['Commits', 'GitHub Distance in Releases', 'Avg Lines', 'Git Resolution Time',
                          'Comments in JIRA', 'Total Deletions', 'Total Insertions', 'Avg Files', 'Change Log Size',
                          'Number of Reopens']

    nominal_features = ['Git Repository']

    figure, axes = plt.subplots(1, 1)
    issues_dataframe['Git Repository'].value_counts(normalize=True).plot(kind='bar', ax=axes)
    # plt.show()

    issues_dataframe, encoded_priorities = prepare_for_training(issues_dataframe, class_label, numerical_features,
                                                                nominal_features)

    issues_train, issues_test, priority_train, priority_test = train_test_split(issues_dataframe,
                                                                                encoded_priorities,
                                                                                test_size=0.2, random_state=0)

    print len(issues_train.index), " issues on the train set."

    issues_train_std, issues_test_std = escale_features(numerical_features, issues_train, issues_test)

    logit_classifier = select_features_l1(issues_train_std, priority_train, issues_test_std, priority_test)
    knn_classifier = sequential_feature_selection(issues_train_std, priority_train, issues_test_std, priority_test)
    forest_classifier = feature_importance_with_forest(issues_train, priority_train, issues_test, priority_test)

    train_and_predict(knn_classifier, original_dataframe, issues_dataframe, encoded_priorities, class_label,
                      numerical_features,
                      nominal_features)
