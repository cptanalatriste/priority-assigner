"""
This modules builds a priority assigner considering possible inflated issues in the training set
"""

import preprocessing
import assigner
import selector
import tuning

from sklearn.cross_validation import train_test_split


def prepare_for_classification(dataframe, scaling_data=None):
    """
    Produces a standarized issue dataframe and its corresponding priority series.
    :param dataframe: Complete dataset.
    :param scaling_data: Training for the standarization process.
    :return: Standarized issues and priorities.
    """
    issues, priorities = preprocessing.encode_and_split(dataframe,
                                                        assigner.CLASS_LABEL,
                                                        assigner.NUMERICAL_FEATURES,
                                                        [])

    if scaling_data is not None:
        _, issues_std = preprocessing.escale_numerical_features(assigner.NUMERICAL_FEATURES,
                                                                scaling_data,
                                                                issues)
    else:
        issues_std, _ = preprocessing.escale_numerical_features(assigner.NUMERICAL_FEATURES,
                                                                issues)

    return issues, issues_std, priorities


def get_training_datasets(raw_dataframe):
    """
    Returns the training and test datasets without filtering only valid data.
    :param raw_dataframe: Original, unfiltered dataframe.
    :return: Train and test sets, with feautures, standarized features and encoded priorities.
    """
    unfiltered_dataframe = preprocessing.filter_issues_dataframe(raw_dataframe, priority_changer=False)
    train, test = train_test_split(unfiltered_dataframe, test_size=0.2, random_state=0)
    issues_train, issues_train_std, priorities_train = prepare_for_classification(train, None)
    issues_test, issues_test_std, priorities_test = prepare_for_classification(test, issues_train)

    return issues_train, issues_train_std, priorities_train, issues_test, issues_test_std, priorities_test


def algorithm_analysis():
    """
    Prepares a classifier based on non-validated data, and evaluates its performance in the valdated
    portion of the dataset.
    :return: Dictionary of classifiers
    """

    raw_dataframe = preprocessing.load_original_dataframe()
    issues_train, issues_train_std, priorities_train, issues_test, issues_test_std, priorities_test = get_training_datasets(
        raw_dataframe)

    unfiltered_dataframe = preprocessing.filter_issues_dataframe(raw_dataframe, priority_changer=False)
    filtered_dataframe = preprocessing.filter_issues_dataframe(raw_dataframe, priority_changer=True)

    train, test = train_test_split(unfiltered_dataframe, test_size=0.2, random_state=0)
    test_valid = preprocessing.filter_issues_dataframe(test, priority_changer=True)
    issues_test_valid, issues_test_valid_std, priorities_test_valid = prepare_for_classification(test_valid,
                                                                                                 issues_train)
    issues_valid, issues_valid_std, priorities_valid = prepare_for_classification(filtered_dataframe, issues_train)

    results = []
    estimators = {}
    for algorithm, grid_search in selector.get_algorithms():
        training_set = issues_train_std
        test_set = issues_test_std
        test_valid_set = issues_test_valid_std
        valid_set = issues_valid_std

        print "Current algorithm: ", algorithm

        if algorithm == "RandomForest":
            training_set = issues_train
            test_set = issues_test
            test_valid_set = issues_test_valid
            valid_set = issues_valid

        optimal_estimator, best_params = tuning.parameter_tuning(grid_search, training_set,
                                                                 priorities_train)
        estimators[algorithm] = optimal_estimator

        result = selector.analyse_performance(optimal_estimator, best_params, grid_search, algorithm, training_set,
                                              priorities_train,
                                              test_set,
                                              priorities_test)

        print "Evaluating on the valid portion of the test dataset ..."

        assigner.evaluate_performance(prefix=algorithm, classifier=optimal_estimator,
                                      issues_test_std=test_valid_set, priority_test=priorities_test_valid)

        print "Evaluating on the complete valid dataset ..."

        assigner.evaluate_performance(prefix=algorithm, classifier=optimal_estimator,
                                      issues_test_std=valid_set, priority_test=priorities_valid)

        if result:
            results.append(result + ("ALL", len(unfiltered_dataframe.index)))

    selector.write_results("all_experiment_results.csv", results)
    return estimators


def main():
    """
    Initial execution point.
    :return: None.
    """

    best_forest = algorithm_analysis()["RandomForest"]

    raw_dataframe = preprocessing.load_original_dataframe()
    issues_train, issues_train_std, priorities_train, issues_test, issues_test_std, priorities_test = get_training_datasets(
        raw_dataframe)

    assigner.feature_importance_with_forest(best_forest, issues_train, priorities_train, issues_test, priorities_test)


if __name__ == "__main__":
    main()
