"""
This module performs experiments to select a classification algorithm.
"""
import sys
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
                try:
                    mean_cv, std_cv = tuning.nested_cross_validation(grid_search, issues_train_std, priority_train)
                    optimal_estimator, best_params = tuning.parameter_tuning(grid_search, issues_train_std,
                                                                             priority_train)

                    train_accuracy, test_accuracy = assigner.evaluate_performance(algorithm, optimal_estimator,
                                                                                  issues_train_std,
                                                                                  priority_train, issues_test_std,
                                                                                  priority_test)

                    print ' *** algorithm: ', algorithm, " mean_cv: ", mean_cv, " std_cv: ", std_cv, " best_params: ", \
                        best_params, " train_accuracy: ", train_accuracy, " test_accuracy: ", test_accuracy
                except:
                    print "!!!!!!  An error ocurred while applying ", algorithm, " to repository ", repository
                    trace = traceback.print_exc()
                    print trace


        else:
            print "Issues corresponding to repository ", repository, " are not enough for analysis."


if __name__ == "__main__":
    main()
