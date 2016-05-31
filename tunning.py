"""
This modules deals with parameter tunning for the Priority Assigner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import learning_curve

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.learning_curve import learning_curve

import assigner


def pipeline_with_PCA(issues_train_std=None, priority_train=None, issues_test_std=None, priority_test=None):
    """
    Defines a pipeline with PCA for a Logistic Regression.
    :param issues_train_std: Issues in training.
    :param priority_train: Priorities in training.
    :param issues_test_std: Issues in test.
    :param priority_test: Priorities in test.
    :return: Pipeline with the classifier.
    """
    lregression_pipeline = Pipeline([
        ('pca', PCA(n_components=2)),
        ('clf', LogisticRegression(random_state=1))])

    lregression_pipeline.fit(issues_train_std, priority_train)

    assigner.evaluate_performance('LOGIT-PCA', lregression_pipeline, issues_train_std, priority_train, issues_test_std,
                                  priority_test)

    return lregression_pipeline


def cross_validation(classifier, issues_train_std, priority_train):
    """
    Applies a stratified k-fold cross validation over the train-set.
    :param classifier: Classifier.
    :param issues_train_std: Standarized train-set
    :param priority_train: Labels for the train-set
    :return: None.
    """
    print issues_train_std.head()

    k_fold = StratifiedKFold(y=priority_train, n_folds=10, random_state=1)
    scores = []

    for fold, (train, test) in enumerate(k_fold):
        fold_issues_train = issues_train_std.iloc[train]
        fold_priorities_train = priority_train.iloc[train]

        fold_issues_test = issues_train_std.iloc[test]
        fold_priorities_test = priority_train.iloc[test]

        classifier.fit(fold_issues_train, fold_priorities_train)
        score = classifier.score(fold_issues_test, fold_priorities_test)
        scores.append(score)

        print "Fold ", fold + 1, ": Class distribution: ", np.unique(fold_priorities_train,
                                                                     return_counts=True), " Score: ", score

    cv_mean = np.mean(scores)
    cv_std = np.std(scores)

    print "CV Accuracy: Mean ", cv_mean, " Std: ", cv_std


def validation_curve(estimator=None, issues_train=None, priority_train=None):
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=issues_train,
                                                            y=priority_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    _, _ = plt.subplots(figsize=(2.5, 2.5))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def main():
    """
    Initial execution point.
    :return: None
    """
    original_dataframe = pd.read_csv(assigner.CSV_FILE)
    filtered_dataframe = assigner.filter_issues_dataframe(original_dataframe)

    issues, priorities = assigner.prepare_for_training(filtered_dataframe, assigner.CLASS_LABEL,
                                                       assigner.NUMERICAL_FEATURES, assigner.NOMINAL_FEATURES)

    issues_train, issues_test, priority_train, priority_test = train_test_split(issues, priorities)
    issues_train_std, issues_test_std = assigner.escale_numerical_features(assigner.NUMERICAL_FEATURES, issues_train,
                                                                           issues_test)

    lr_pca_pipeline = pipeline_with_PCA(issues_train_std, priority_train, issues_test_std, priority_test)
    cross_validation(lr_pca_pipeline, issues_train_std, priority_train)

    lregression_l2 = Pipeline([('clf', LogisticRegression(penalty='l2',
                                                          random_state=0))])
    validation_curve(lregression_l2, issues_train_std, priority_train)

    lregression_l2.fit(issues_train_std, priority_train)
    assigner.evaluate_performance("LOGIT-L2", lregression_l2, issues_train_std, priority_train, issues_test_std,
                                  priority_test)



if __name__ == "__main__":
    main()
