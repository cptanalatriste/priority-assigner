"""
This module performs experiments to select a classification algorithm.
"""

import pandas as pd
import numpy as np
import traceback
import winsound

import assigner
import tuning
import preprocessing

from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

SCORING = 'f1_macro'


def get_algorithms():
    """
    Returns the algorithms to evaluate as grid search instances.
    :return: List of grid search instances.
    """
    cv = 5
    n_jobs = 1

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    small_param_range = [0.0001, 1.0, 10.0]

    return [

        ("LogisticRegression", GridSearchCV(estimator=LogisticRegression(),
                                            param_grid=[{'penalty': ['l1', 'l2'],
                                                         'C': param_range,
                                                         'class_weight': ['balanced']}],
                                            scoring=SCORING,
                                            cv=cv,
                                            n_jobs=n_jobs), True),
        ("KNeighbors", GridSearchCV(estimator=KNeighborsClassifier(),
                                    param_grid=[{'n_neighbors': np.arange(1, 20, 1),
                                                 'weights': ['uniform', 'distance']}],
                                    scoring=SCORING,
                                    cv=cv,
                                    n_jobs=n_jobs), True),
        ("RandomForest", GridSearchCV(estimator=RandomForestClassifier(random_state=0,
                                                                       n_jobs=-1),
                                      param_grid=[{'n_estimators': np.arange(1, 100, 10),
                                                   'max_depth': np.arange(1, 100, 20)}],
                                      scoring=SCORING,
                                      cv=cv,
                                      n_jobs=n_jobs), False),
        ("SVM", GridSearchCV(estimator=SVC(),
                             param_grid=[{'C': [0.0001, 1.0, 10.0],
                                          'kernel': ['linear'],
                                          'class_weight': ['balanced']},
                                         {'C': param_range,
                                          'gamma': param_range,
                                          'kernel': ['rbf'],
                                          'class_weight': ['balanced']}],
                             scoring=SCORING,
                             cv=cv,
                             n_jobs=n_jobs), True)

    ]


def run_algorithm_analysis(issues_train=None, issues_train_std=None, labels_train=None, issues_test=None,
                           issues_test_std=None, labels_test=None,
                           repository=None, issues_found=None):
    """
    Executes the algorithm list against a train-test dataset.

    :param issues_train_std: Issues for training.
    :param labels_train: Priorities in the training set.
    :param issues_test_std: Issues for testing.
    :param labels_test: Priorities for the test set.
    :param repository: Name of the repository.
    :param issues_found: Issues contained in the repository.
    :return: List of tuples with the information.
    """
    results = []
    for algorithm, grid_search, standarize in get_algorithms():

        print "Executing ", algorithm, " over ", repository, " dataset ..."

        train_dataset = issues_train_std
        test_dataset = issues_test_std

        if not standarize:
            train_dataset = issues_train
            test_dataset = issues_test

        optimal_estimator, best_params = tuning.parameter_tuning(grid_search, train_dataset,
                                                                 labels_train)

        result = analyse_performance(optimal_estimator=optimal_estimator, best_params=best_params,
                                     grid_search=grid_search, algorithm=algorithm, train_dataset=train_dataset,
                                     labels_train=labels_train,
                                     test_dataset=test_dataset,
                                     labels_test=labels_test)
        if result:
            results.append(result + (repository, issues_found))

    return results


def analyse_performance(optimal_estimator=None, best_params=None, grid_search=None, algorithm=None,
                        train_dataset=None, labels_train=None,
                        test_dataset=None,
                        labels_test=None):
    """
    Returns performance metrics for a classification algorithm, after tuning its parameters through grid search.

    :param optimal_estimator: Estimator to evaluate
    :param grid_search: Grid Search, to evaluate nested cross validation.
    :param best_params : Estimator configuration.
    :param algorithm: Algorithm description.
    :param train_dataset: Issues in train.
    :param labels_train: Priorities in train.
    :param test_dataset: Issues in test.
    :param labels_test: Priorities in test.
    :return: A tuple with performance metrics.
    """
    try:
        mean_cv, std_cv = tuning.nested_cross_validation(grid_search, train_dataset, labels_train,
                                                         SCORING)

        train_accuracy, test_accuracy, test_f1score, f1score_class, support_per_class = assigner.evaluate_performance(
            algorithm, optimal_estimator,
            train_dataset,
            labels_train, test_dataset,
            labels_test)

        return (
            (algorithm, mean_cv, std_cv, best_params, train_accuracy,
             test_accuracy,
             test_f1score, f1score_class[1], f1score_class[2], f1score_class[3],
             f1score_class[4], f1score_class[5], support_per_class[1], support_per_class[2],
             support_per_class[3],
             support_per_class[4], support_per_class[5]))
    except ValueError as e:
        print "!!!!!!  An error ocurred while applying ", algorithm
        trace = traceback.print_exc()
        print trace

    return None


def write_results(file_name, results):
    """
    Writes the experiment results into a CSV file.
    :param results: List with tuples, each containing experiment execution.
    :return: None
    """

    print "Writing results to ", file_name
    results_dataframe = pd.DataFrame(data=results,
                                     columns=["Algorithm", "CV Mean", " CV STD",
                                              "Best configuration",
                                              "Train accuracy", "Test accuracy", " Test f1-score",
                                              "Test f1-score Pri 1", "Test f1-score Pri 2", "Test f1-score Pri 3",
                                              "Test f1-score Pri 4", "Test f1-score Pri 5", "Test Support Pri 1",
                                              "Test Support Pri 2", "Test Support Pri 3",
                                              "Test Support Pri 4", "Test Support Pri 5", "Repository",
                                              "Total Issues"])

    results_dataframe.to_csv(file_name, index=False)


def get_all_repositories(dataframe):
    """
    Returns all the repositories available on a dataframe
    :param dataframe: The dataframe
    :return: Repository field values.
    """
    return dataframe['Git Repository'].unique()


def main():
    """
    Initial execution point
    :return: None
    """

    original_dataframe = preprocessing.load_original_dataframe()
    repositories = get_all_repositories(original_dataframe)

    results = []

    for repository in repositories:
        print "Working on repository ", repository, " ..."

        project_dataframe = preprocessing.filter_issues_dataframe(original_dataframe, repository=repository)

        # Threslhold taking into account considering the scikit-learn cheat sheet
        # http://scikit-learn.org/stable/tutorial/machine_learning_map/

        minimum_threshold = 50
        issues_found = len(project_dataframe.index)
        print issues_found, " issues found on repository ", repository

        if issues_found > minimum_threshold:

            # The Git Repository feauture is not needed since it is filtered.
            nominal_features = []
            issues, priorities = preprocessing.encode_and_split(project_dataframe, assigner.CLASS_LABEL,
                                                                assigner.NUMERICAL_FEATURES, nominal_features)

            train_test = preprocessing.train_test_encode(repository, issues, priorities)
            if train_test:
                issues_train_std, priority_train, issues_test_std, priority_test = train_test
                result = run_algorithm_analysis(issues_train_std, priority_train, issues_test_std, priority_test,
                                                repository, issues_found)

                if result:
                    results.extend(result)

        else:
            print "Issues corresponding to repository ", repository, " are not enough for analysis."

    write_results("project_experiment_results.csv", results)


if __name__ == "__main__":
    main()
    winsound.Beep(2500, 1000)
