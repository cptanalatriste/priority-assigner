"""
This module performs experiments to select a classification algorithm.
"""

import pandas as pd
import traceback

import assigner
import tuning

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import LogisticRegression


def get_algorithms():
    """
    Returns the algorithms to evaluate as grid search instances.
    :return: List of grid search instances.
    """
    scoring = 'accuracy'
    cv = 5
    n_jobs = 1

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    return [("LogisticRegression", GridSearchCV(estimator=LogisticRegression(),
                                                param_grid=[{'penalty': ['l1', 'l2'],
                                                             'C': param_range}],
                                                scoring=scoring,
                                                cv=cv,
                                                n_jobs=n_jobs))]


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

        minimum_threshold = 10
        issues_found = len(project_dataframe.index)
        print issues_found, " issues found on repository ", repository

        if issues_found > minimum_threshold:

            # The Git Repository feauture is not needed since it is filtered.
            nominal_features = []
            issues, priorities = assigner.prepare_for_training(project_dataframe, assigner.CLASS_LABEL,
                                                               assigner.NUMERICAL_FEATURES, nominal_features)

            issues_train, issues_test, priority_train, priority_test = train_test_split(issues, priorities)

            issues_train_std, issues_test_std = assigner.escale_numerical_features(assigner.NUMERICAL_FEATURES,
                                                                                   issues_train,
                                                                                   issues_test)

            for algorithm, grid_search in get_algorithms():
                mean_cv, std_cv = tuning.nested_cross_validation(grid_search, issues_train_std, priority_train)
                optimal_estimator, best_params = tuning.parameter_tuning(grid_search, issues_train_std,
                                                                         priority_train)

                train_accuracy, test_accuracy, test_f1score, f1score_class, support_per_class = assigner.evaluate_performance(
                    algorithm, optimal_estimator,
                    issues_train_std,
                    priority_train, issues_test_std,
                    priority_test)

                results.append((algorithm, repository, mean_cv, std_cv, best_params, train_accuracy, test_accuracy,
                                test_f1score, f1score_class[1], f1score_class[2], f1score_class[3],
                                f1score_class[4], f1score_class[5], support_per_class[1], support_per_class[2],
                                support_per_class[3],
                                support_per_class[4], support_per_class[5]))
        else:
            print "Issues corresponding to repository ", repository, " are not enough for analysis."

    file_name = "experiment_results.csv"

    print "Writing results to ", file_name
    results_dataframe = pd.DataFrame(data=results,
                                     columns=["Algorithm", "Repository", "CV Mean", " CV STD", "Best configuration",
                                              "Train accuracy", "Test accuracy", " Test f1-score",
                                              "Test f1-score Pri 1", "Test f1-score Pri 2", "Test f1-score Pri 3",
                                              "Test f1-score Pri 4", "Test f1-score Pri 5", "Test Support Pri 1",
                                              "Test Support Pri 2", "Test Support Pri 3",
                                              "Test Support Pri 4", "Test Support Pri 5"])

    results_dataframe.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()
