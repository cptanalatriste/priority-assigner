"""
This module builds a predictor for the priority field for a bug report
"""

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

import fselect
import preprocessing


def evaluate_performance(prefix=None, classifier=None, issues_train=None, priority_train=None,
                         issues_test=None, priority_test=None):
    """
    Calculates performance metrics for a classifier.
    :param prefix: A prefix, for identifying the classifier.
    :param classifier: The classifier, previously fitted.
    :param issues_train: Train features.
    :param priority_train: Train class.
    :param issues_test: Test features.
    :param priority_test: Test class.
    :return: Train accuracy , Test accuracy, Test weighted-f1 and F1 score per class.
    """

    train_accuracy = None
    if issues_train is not None and priority_train is not None:
        train_accuracy = classifier.score(issues_train, priority_train)
        print prefix, ': Training accuracy ', train_accuracy
        train_predictions = classifier.predict(issues_train)

        print prefix, " :TRAIN DATA SET"
        print classification_report(y_true=priority_train, y_pred=train_predictions)

    test_accuracy = classifier.score(issues_test, priority_test)
    print prefix, ': Test accuracy ', test_accuracy

    test_predictions = classifier.predict(issues_test)

    test_kappa = cohen_kappa_score(priority_test, test_predictions)
    print prefix, ": Test Kappa: ", test_kappa

    print prefix, " :TEST DATA SET"
    print classification_report(y_true=priority_test, y_pred=test_predictions)

    labels = np.sort(np.unique(np.concatenate((priority_test.values, test_predictions))))
    test_f1_score = f1_score(y_true=priority_test, y_pred=test_predictions, average='weighted')
    f1_scores = f1_score(y_true=priority_test, y_pred=test_predictions, average=None)

    f1_per_class = {label: score for label, score in zip(labels, f1_scores)}

    all_scores = precision_recall_fscore_support(y_true=priority_test, y_pred=test_predictions, average=None)

    support_index = 3
    support_per_class = {label: support for label, support in zip(labels, all_scores[support_index])}

    return train_accuracy, test_accuracy, test_f1_score, defaultdict(lambda: 0, f1_per_class), \
           defaultdict(lambda: 0, support_per_class)


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

    print 'Intercept: ', logistic_regression.intercept_
    print 'Coefficient: ', logistic_regression.coef_

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

    print "Applying sequential feature selection..."
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

    figure, axes = plt.subplots(1, 1)
    plt.plot(num_features, feature_selector.scores_, marker='o')
    plt.ylim([0.4, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()

    plt.show()

    knn_classifier.fit(issues_train_std, priority_train)
    evaluate_performance("KNN-5NEIGH", knn_classifier, issues_train_std, priority_train, issues_test_std, priority_test)

    new_train = issues_train_std.iloc[:, optimal_subset]
    new_test = issues_test_std.iloc[:, optimal_subset]

    knn_classifier.fit(new_train, priority_train)
    evaluate_performance("KNN-OPT", knn_classifier, new_train, priority_train, new_test, priority_test)

    return knn_classifier


def feature_importance_with_forest(rforest_classifier, issues_train, priority_train, issues_test, priority_test):
    """
    Assess feature importance using a Random Forest.
    :param rforest_classifier: An already fitted classifier.
    :param issues_train: Train features.
    :param priority_train: Train classes.
    :param issues_test: Test features.
    :param priority_test: Test classes.
    :return: None
    """
    importances = rforest_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    for column_index in range(len(issues_train.columns)):
        print column_index + 1, ") ", issues_train.columns[column_index], " ", importances[indices[column_index]]

    figure, axes = plt.subplots(1, 1)
    plt.title('Feature importance')
    plt.bar(range(len(issues_train.columns)), importances[indices], color='lightblue', align='center')
    plt.xticks(range(len(issues_train.columns)), issues_train.columns, rotation=90)
    plt.xlim([-1, len(issues_train.columns)])
    plt.tight_layout()
    plt.show()

    evaluate_performance("FOREST", rforest_classifier, issues_train, priority_train, issues_test, priority_test)

    print "Selecting important features ..."
    select = SelectFromModel(rforest_classifier, threshold=0.05, prefit=True)

    train_selected = select.transform(issues_train)
    test_selected = select.transform(issues_test)

    rforest_classifier.fit(train_selected, priority_train)
    evaluate_performance("FOREST-IMPORTANT", rforest_classifier, train_selected, priority_train, test_selected,
                         priority_test)


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
    issues_dataframe, encoded_priorities = preprocessing.encode_and_split(temp_dataframe, class_label,
                                                                          numerical_features,
                                                                          nominal_features)

    training_std, original_std = preprocessing.escale_numerical_features(numerical_features, training_dataframe,
                                                                         issues_dataframe)

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


def main():
    original_dataframe = preprocessing.load_original_dataframe()
    issues_dataframe = preprocessing.filter_issues_dataframe(original_dataframe)

    # Plotting projects
    figure, axes = plt.subplots(1, 1)
    issues_dataframe['Git Repository'].value_counts(normalize=True).plot(kind='bar', ax=axes)
    plt.show()

    issues_dataframe, encoded_priorities = preprocessing.encode_and_split(issues_dataframe, preprocessing.CLASS_LABEL,
                                                                          preprocessing.NUMERICAL_FEATURES,
                                                                          preprocessing.NOMINAL_FEATURES)

    # Plotting priorities

    figure, axes = plt.subplots(1, 1)
    encoded_priorities.value_counts(normalize=True, sort=True).plot(kind='bar', ax=axes)
    plt.show()

    issues_train, issues_test, priority_train, priority_test = train_test_split(issues_dataframe,
                                                                                encoded_priorities,
                                                                                test_size=0.2, random_state=0)

    print len(issues_train.index), " issues on the train set."

    issues_train_std, issues_test_std = preprocessing.escale_numerical_features(preprocessing.NUMERICAL_FEATURES,
                                                                                issues_train,
                                                                                issues_test)

    logit_classifier = select_features_l1(issues_train_std, priority_train, issues_test_std, priority_test)
    knn_classifier = sequential_feature_selection(issues_train_std, priority_train, issues_test_std, priority_test)

    print "Building Random Forest Classifier ..."
    rforest_classifier = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    rforest_classifier.fit(issues_train, priority_train)
    forest_classifier = feature_importance_with_forest(issues_train, priority_train, issues_test, priority_test)

    rforest_classifier = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

    train_and_predict(rforest_classifier, original_dataframe, issues_dataframe, encoded_priorities,
                      preprocessing.CLASS_LABEL,
                      preprocessing.NUMERICAL_FEATURES,
                      preprocessing.NOMINAL_FEATURES)


if __name__ == "__main__":
    main()
