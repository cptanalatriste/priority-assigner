"""
This modules builds a priority assigner considering possible inflated issues in the training set
"""

import assigner
import selector
import tuning

from sklearn.cross_validation import train_test_split


def get_filtered_dataset(raw_dataframe=None, issues_for_std=None):
    """
    Generates a dataset for prediction taking into account only issues with priority changes.
    :param raw_dataframe: Unfiltered dataframe
    :param issues_for_std: The issues dataset used for training, for standarization purposes.
    :return: Standarized issues and its corresponding priorities.
    """
    filtered_dataframe = assigner.filter_issues_dataframe(raw_dataframe, priority_changer=True)
    filtered_issues, filtered_priorities = assigner.prepare_for_training(filtered_dataframe, assigner.CLASS_LABEL,
                                                                         assigner.NUMERICAL_FEATURES, [])

    _, filtered_issues_std = assigner.escale_numerical_features(assigner.NUMERICAL_FEATURES, issues_for_std,
                                                                filtered_issues)

    return filtered_issues_std, filtered_priorities


def main():
    """
    Initial execution point.
    :return: None.
    """

    raw_dataframe = assigner.load_original_dataframe()
    original_dataframe = assigner.filter_issues_dataframe(raw_dataframe, priority_changer=False)

    # Excluding the project membership feature.
    issues, priorities = assigner.prepare_for_training(original_dataframe, assigner.CLASS_LABEL,
                                                       assigner.NUMERICAL_FEATURES, [])

    issues_train, issues_test, priority_train, priority_test = train_test_split(issues, priorities,
                                                                                test_size=0.2,
                                                                                stratify=priorities,
                                                                                random_state=0)

    issues_train_std, issues_test_std = assigner.escale_numerical_features(assigner.NUMERICAL_FEATURES,
                                                                           issues_train,
                                                                           issues_test)

    filtered_issues_std, filtered_priorities = get_filtered_dataset(raw_dataframe, issues_train)

    # TODO All these previous steps should be moved to their own module.

    results = []
    for algorithm, grid_search in selector.get_algorithms():
        optimal_estimator, best_params = tuning.parameter_tuning(grid_search, issues_train_std,
                                                                 priority_train)

        result = selector.analyse_performance(optimal_estimator, best_params, grid_search, algorithm, issues_train_std,
                                              priority_train,
                                              issues_test_std,
                                              priority_test)

        print "Evaluating on the filtered dataset ..."

        assigner.evaluate_performance(prefix=algorithm, classifier=optimal_estimator,
                                      issues_test_std=filtered_issues_std, priority_test=filtered_priorities)

        if result:
            results.append(result + ("ALL", len(original_dataframe.index)))

    selector.write_results("all_experiment_results.csv", results)


if __name__ == "__main__":
    main()
