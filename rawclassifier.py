"""
This modules builds a priority assigner considering possible inflated issues in the training set
"""

import preprocessing
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

    raw_dataframe = preprocessing.load_original_dataframe()
    unfiltered_dataframe = preprocessing.filter_issues_dataframe(raw_dataframe, priority_changer=False)
    filtered_dataframe = preprocessing.filter_issues_dataframe(raw_dataframe, priority_changer=True)

    train, test = train_test_split(unfiltered_dataframe, test_size=0.2, random_state=0)
    test_valid = preprocessing.filter_issues_dataframe(test, priority_changer=True)

    issues_train, priorities_train = preprocessing.encode_and_split(train,
                                                                    assigner.CLASS_LABEL,
                                                                    assigner.NUMERICAL_FEATURES,
                                                                    [])
    issues_test, priorities_test = preprocessing.encode_and_split(test,
                                                                  assigner.CLASS_LABEL,
                                                                  assigner.NUMERICAL_FEATURES,
                                                                  [])

    issues_test_valid, priorities_test_valid = preprocessing.encode_and_split(test_valid,
                                                                              assigner.CLASS_LABEL,
                                                                              assigner.NUMERICAL_FEATURES,
                                                                              [])
    issues_valid, priorities_valid = preprocessing.encode_and_split(filtered_dataframe,
                                                                    assigner.CLASS_LABEL,
                                                                    assigner.NUMERICAL_FEATURES,
                                                                    [])

    # Feature scaling, using the training data as a reference
    issues_train_std, issues_test_std = preprocessing.escale_numerical_features(assigner.NUMERICAL_FEATURES,
                                                                                issues_train, issues_test)
    _, issues_test_valid_std = preprocessing.escale_numerical_features(assigner.NUMERICAL_FEATURES, issues_train,
                                                                       issues_test_valid)
    _, issues_valid_std = preprocessing.escale_numerical_features(assigner.NUMERICAL_FEATURES, issues_train,
                                                                  issues_valid)

    print "len(issues_train_std.index) ", len(issues_train_std.index)
    print "len(issues_test_std.index) ", len(issues_test_std.index)
    print "len(issues_test_valid_std.index) ", len(issues_test_valid_std.index)
    print "len(issues_valid_std.index) ", len(issues_valid_std.index)

    # TODO All these previous steps should be moved to their own module.

    results = []
    for algorithm, grid_search in selector.get_algorithms():
        optimal_estimator, best_params = tuning.parameter_tuning(grid_search, issues_train_std,
                                                                 priorities_train)

        result = selector.analyse_performance(optimal_estimator, best_params, grid_search, algorithm, issues_train_std,
                                              priorities_train,
                                              issues_test_std,
                                              priorities_test)

        print "Evaluating on the valid portion of the test dataset ..."

        assigner.evaluate_performance(prefix=algorithm, classifier=optimal_estimator,
                                      issues_test_std=issues_test_valid, priority_test=priorities_test_valid)

        print "Evaluating on the complete valid dataset ..."

        assigner.evaluate_performance(prefix=algorithm, classifier=optimal_estimator,
                                      issues_test_std=issues_valid_std, priority_test=priorities_valid)

        if result:
            results.append(result + ("ALL", len(unfiltered_dataframe.index)))

    selector.write_results("all_experiment_results.csv", results)


if __name__ == "__main__":
    main()
