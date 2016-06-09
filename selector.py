"""
This module performs experiments to select a classification algorithm.
"""

import pandas as pd
import numpy as np
import traceback
import winsound

import assigner
import tuning

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

SCORING = 'f1_weighted'


def get_algorithms():
    """
    Returns the algorithms to evaluate as grid search instances.
    :return: List of grid search instances.
    """
    cv = 5
    n_jobs = 1

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    return [

        ("LogisticRegression", GridSearchCV(estimator=LogisticRegression(),
                                            param_grid=[{'penalty': ['l1', 'l2'],
                                                         'C': param_range}],
                                            scoring=SCORING,
                                            cv=cv,
                                            n_jobs=n_jobs)),
        ("KNeighbors", GridSearchCV(estimator=KNeighborsClassifier(),
                                    param_grid=[{'n_neighbors': np.arange(1, 20, 1),
                                                 'weights': ['uniform', 'distance']}],
                                    scoring=SCORING,
                                    cv=cv,
                                    n_jobs=n_jobs)),
        ("RandomForest", GridSearchCV(estimator=RandomForestClassifier(random_state=0,
                                                                       n_jobs=-1),
                                      param_grid=[{'n_estimators': np.arange(1, 100, 10)}],
                                      scoring=SCORING,
                                      cv=cv,
                                      n_jobs=n_jobs)),
        ("SVM", GridSearchCV(estimator=SVC(),
                             param_grid=[{'C': param_range,
                                          'kernel': ['linear']},
                                         # {'C': param_range,
                                         #  'gamma': param_range,
                                         #  'kernel': ['rbf']}
                                         ],
                             scoring=SCORING,
                             cv=cv,
                             n_jobs=n_jobs))
    ]


def analyse_performance(optimal_estimator=None, best_params=None, grid_search=None, algorithm=None,
                        issues_train_std=None, priority_train=None,
                        issues_test_std=None,
                        priority_test=None):
    """
    Returns performance metrics for a classification algorithm, after tuning its parameters through grid search.

    :param optimal_estimator: Estimator to evaluate
    :param best_params : Estimator configuration.
    :param algorithm: Algorithm description.
    :param issues_train_std: Issues in train.
    :param priority_train: Priorities in train.
    :param issues_test_std: Issues in test.
    :param priority_test: Priorities in test.
    :return: A tuple with performance metrics.
    """
    try:
        mean_cv, std_cv = tuning.nested_cross_validation(grid_search, issues_train_std, priority_train,
                                                         SCORING)

        train_accuracy, test_accuracy, test_f1score, f1score_class, support_per_class = assigner.evaluate_performance(
            algorithm, optimal_estimator,
            issues_train_std,
            priority_train, issues_test_std,
            priority_test)

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


def split_train_test(repository="", issues=None, priorities=None):
    """
    Performs the train-test split using stratified sampling, and also normalized the numerical features.

    :param repository: Repository name.
    :param issues: Issues
    :param priorities: Priorities.
    :return: Normalized train issues, train priorities, normalized test issues, test priorities.
    """
    try:
        print "Priorities distribution:\n ", priorities.value_counts()

        issues_train, issues_test, priority_train, priority_test = train_test_split(issues, priorities,
                                                                                    test_size=0.2,
                                                                                    stratify=priorities,
                                                                                    random_state=0)

        issues_train_std, issues_test_std = assigner.escale_numerical_features(assigner.NUMERICAL_FEATURES,
                                                                               issues_train,
                                                                               issues_test)

        return issues_train_std, priority_train, issues_test_std, priority_test

    except ValueError as e:
        print "!!!!!!  An error ocurred while splitting ", repository
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
                                              "Test Support Pri 4", "Test Support Pri 5", "Repository", "Total Issues"])

    results_dataframe.to_csv(file_name, index=False)


def run_algorithm_analysis(issues_train_std=None, priority_train=None, issues_test_std=None, priority_test=None,
                           repository=None, issues_found=None):
    """
    Executes the algorithm list against a train-test dataset.

    :param issues_train_std: Issues for training.
    :param priority_train: Priorities in the training set.
    :param issues_test_std: Issues for testing.
    :param priority_test: Priorities for the test set.
    :param repository: Name of the repository.
    :param issues_found: Issues contained in the repository.
    :return: List of tuples with the information.
    """
    results = []
    for algorithm, grid_search in get_algorithms():

        print "Executing ", algorithm, " over ", repository, " dataset ..."

        optimal_estimator, best_params = tuning.parameter_tuning(grid_search, issues_train_std,
                                                                 priority_train)

        result = analyse_performance(optimal_estimator, best_params, algorithm, issues_train_std, priority_train,
                                     issues_test_std,
                                     priority_test)
        if result:
            results.append(result + (repository, issues_found))

    return results


def main():
    """
    Initial execution point
    :return: None
    """

    original_dataframe = assigner.load_original_dataframe()
    repositories = original_dataframe['Git Repository'].unique()

    results = []

    for repository in repositories:
        print "Working on repository ", repository, " ..."

        project_dataframe = assigner.filter_issues_dataframe(original_dataframe, repository=repository)

        # Threslhold taking into account considering the scikit-learn cheat sheet
        # http://scikit-learn.org/stable/tutorial/machine_learning_map/

        minimum_threshold = 50
        issues_found = len(project_dataframe.index)
        print issues_found, " issues found on repository ", repository

        if issues_found > minimum_threshold:

            # The Git Repository feauture is not needed since it is filtered.
            nominal_features = []
            issues, priorities = assigner.prepare_for_training(project_dataframe, assigner.CLASS_LABEL,
                                                               assigner.NUMERICAL_FEATURES, nominal_features)

            train_test = split_train_test(repository, issues, priorities)
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
