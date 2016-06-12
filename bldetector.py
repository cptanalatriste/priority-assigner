"""
This Module tries to build a classifier to identify Blocker issues.
"""
import traceback

import preprocessing
import selector


def execute_analysis(dataframe, repository, class_label):
    """
    Given a dataframe, it splits in train and test, executes and tunes several classification
    algorithms and shows the execution results.

    :param dataframe: Dataframe containing issue information.
    :param repository: Description of the repository.
    :param class_label: Name of the feature that contains the class
    :return: None
    """
    try:
        issues, labels = preprocessing.encode_and_split(issues_dataframe=dataframe,
                                                        class_label=class_label,
                                                        numerical_features=preprocessing.NUMERICAL_FEATURES,
                                                        nominal_features=[])

        issues_train, issues_train_std, labels_train, issues_test, issues_test_std, labels_test = preprocessing.train_test_encode(
            issues=issues,
            labels=labels)

        results = selector.run_algorithm_analysis(repository=repository,
                                                  issues_train=issues_train,
                                                  issues_train_std=issues_train_std,
                                                  labels_train=labels_train,
                                                  issues_test=issues_test,
                                                  issues_test_std=issues_test_std,
                                                  labels_test=labels_test)
    except ValueError as e:
        print "!!!!!!  An error ocurred while working on ", repository
        trace = traceback.print_exc()
        print trace


def main():
    original_dataframe = preprocessing.load_original_dataframe()
    class_label = 'Blocker'
    # class_label = 'Trivial'
    # class_label = 'Critical'
    # class_label = 'Severe'
    class_label = "Non-Severe"

    print "Using ", class_label, " as the class feature."

    # valid_dataframe = preprocessing.filter_issues_dataframe(original_dataframe)
    # execute_analysis(valid_dataframe, 'VALID', class_label)

    repositories = selector.get_all_repositories(original_dataframe)
    for repository in repositories:
        print "Working on repository ", repository, " ..."

        project_dataframe = preprocessing.filter_issues_dataframe(original_dataframe, repository=repository)
        execute_analysis(project_dataframe, repository, class_label)


if __name__ == "__main__":
    main()
