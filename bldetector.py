"""
This Module tries to build a classifier to identify Blocker issues.
"""

import preprocessing
import selector


def execute_analysis(dataframe, repository, class_label):
    issues, labels = preprocessing.encode_and_split(issues_dataframe=dataframe,
                                                    class_label=class_label,
                                                    numerical_features=preprocessing.NUMERICAL_FEATURES)

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


def main():
    original_dataframe = preprocessing.load_original_dataframe()
    valid_dataframe = preprocessing.filter_issues_dataframe(original_dataframe)

    # execute_analysis(valid_dataframe, 'VALID', 'Blocker')
    execute_analysis(valid_dataframe, 'VALID', 'Trivial')


if __name__ == "__main__":
    main()
