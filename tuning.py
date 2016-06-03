"""
This modules deals with parameter tuning for the Priority Assigner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import learning_curve

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

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


def learning_curve_analysis(estimator=None, issues_train=None, priority_train=None):
    """
    Generates the learning curve for a specific estimator.
    :param estimator: Estimator.
    :param issues_train: Standarized train set.
    :param priority_train: Priorities for train set.
    :return: None.
    """
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


def validation_curve_analysis(estimator=None, param_name=None, param_range=None, issues_train=None,
                              priority_train=None):
    """
    Generates the validation curve for a specific estimator.
    :param estimator: Estimator.
    :param param_name: Name of the parameter.
    :param param_range: Range of the parameters to consider.
    :param issues_train: Train issues.
    :param priority_train: Train priorities.
    :return: None.
    """
    train_scores, test_scores = validation_curve(estimator=estimator, X=issues_train, y=priority_train,
                                                 param_name=param_name, param_range=param_range, cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    _, _ = plt.subplots(figsize=(2.5, 2.5))
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.xlabel('Parameter ' + param_name)
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def parameter_tuning(grid_search=None, issues_train=None, priority_train=None):
    """
    Applies grid search to find optimal parameter configuration.
    :param grid_search: Grid search to run.
    :param issues_train: Train issues.
    :param priority_train: Train priorities.
    :return: Estimator with optimal configuration.
    """
    print "Starting grid search ..."
    grid_search = grid_search.fit(issues_train, priority_train)
    best_params = grid_search.best_params_

    print 'grid_search.best_score_: ', grid_search.best_score_
    print 'grid_search.best_params_: ', best_params

    best_estimator = grid_search.best_estimator_
    return best_estimator, best_params


def nested_cross_validation(grid_search=None, issues_train=None, priority_train=None):
    """
    Applies nested cross-validation, for an specific grid search configuration.
    :param grid_search: Grid Search configuration
    :param issues_train: Train issues.
    :param priority_train: Train priorities.
    :return: Mean as Std of the Cross Validation process.
    """
    print "Starting nested cross-validation ..."
    scores = cross_val_score(grid_search, issues_train, priority_train, scoring='accuracy', cv=5)
    mean_cv, std_cv = np.mean(scores), np.std(scores)
    print 'CV accuracy. Mean: ', np.mean(scores), " Std: ", np.std(scores)

    return mean_cv, std_cv


def main():
    """
    Initial execution point.
    :return: None
    """
    original_dataframe = assigner.load_original_dataframe()
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
    learning_curve_analysis(lregression_l2, issues_train_std, priority_train)

    lregression_l2.fit(issues_train_std, priority_train)
    assigner.evaluate_performance("LOGIT-L2", lregression_l2, issues_train_std, priority_train, issues_test_std,
                                  priority_test)

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    validation_curve_analysis(lregression_l2, 'clf__C', param_range, issues_train_std, priority_train)

    svm_pipeline = Pipeline([('clf', SVC(random_state=1))])
    svm_param_grid = [{'clf__C': param_range,
                       'clf__kernel': ['linear']},
                      {'clf__C': param_range,
                       'clf__gamma': param_range,
                       'clf__kernel': ['rbf']}]

    grid_search = GridSearchCV(estimator=svm_pipeline,
                               param_grid=svm_param_grid,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=-1)

    best_estimator, _ = parameter_tuning(grid_search, issues_train_std, priority_train)
    best_estimator.fit(issues_train_std, priority_train)

    assigner.evaluate_performance("SVM-AFTERGRID", best_estimator, issues_train_std, priority_train, issues_test_std,
                                  priority_test)

    grid_search = GridSearchCV(estimator=svm_pipeline,
                               param_grid=svm_param_grid,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=-1)
    print "SVM"
    nested_cross_validation(grid_search, issues_train_std, priority_train)
    best_svm, _ = parameter_tuning(grid_search, issues_train_std, priority_train)
    assigner.evaluate_performance("SVM-BEST", issues_train_std, priority_train, issues_test_std, priority_test)

    tree_classifier = DecisionTreeClassifier(random_state=0)
    tree_param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]
    grid_search = GridSearchCV(estimator=tree_classifier,
                               param_grid=tree_param_grid,
                               scoring='accuracy',
                               cv=5)

    print "Decission Tree"
    nested_cross_validation(grid_search, issues_train, priority_train)
    best_tree, _ = parameter_tuning(grid_search, issues_train, priority_train)
    assigner.evaluate_performance("TREE-BEST", best_tree, issues_train, priority_train, issues_test,
                                  priority_test)


if __name__ == "__main__":
    main()
