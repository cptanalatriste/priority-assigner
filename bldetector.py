"""
This Module tries to build a classifier to identify Blocker issues.
"""
import traceback
import winsound

import assigner
import preprocessing
import selector

from sklearn.ensemble import RandomForestClassifier


def analyse_forest_features(issues_train=None, priority_train=None, issues_test=None, priority_test=None):
    best_estimators = 31
    best_depth = 61

    rforest_classifier = RandomForestClassifier(n_estimators=best_estimators, max_depth=best_depth, random_state=0,
                                                n_jobs=1)
    rforest_classifier.fit(issues_train, priority_train)
    forest_classifier = assigner.feature_importance_with_forest(rforest_classifier, issues_train, priority_train,
                                                                issues_test,
                                                                priority_test)


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
                                                        nominal_features=[],
                                                        # text_feature=preprocessing.TEXT_FEATURE)
                                                        text_feature=None)

        issues_train, issues_train_std, labels_train, issues_test, issues_test_std, labels_test = preprocessing.train_test_encode(
            issues=issues,
            labels=labels,
            text_feature=None,
            repository=repository)

        # Only for testing
        # analyse_forest_features(issues_train, labels_train, issues_test, labels_test)
        # return

        results = selector.run_algorithm_analysis(repository=repository,
                                                  issues_train=issues_train,
                                                  issues_train_std=issues_train_std,
                                                  labels_train=labels_train,
                                                  issues_test=issues_test,
                                                  issues_test_std=issues_test_std,
                                                  labels_test=labels_test)

        return results

    except ValueError as e:
        print "!!!!!!  An error ocurred while working on ", repository
        trace = traceback.print_exc()
        print trace

        return []


def main():
    consolidated_results = []

    try:
        minimum_records = 50
        class_labels = [
            'Severe'  # , 'Blocker', 'Non-Severe', 'Trivial', 'Critical'
        ]

        original_dataframe = preprocessing.load_original_dataframe()

        valid_dataframe = preprocessing.filter_issues_dataframe(original_dataframe)
        repositories = []
        repositories.append(("VALID", valid_dataframe))

        for repo_name in selector.get_all_repositories(original_dataframe):
            project_dataframe = preprocessing.filter_issues_dataframe(original_dataframe,
                                                                      repository=repo_name)
            repositories.append((repo_name, project_dataframe))

        for class_label in class_labels:
            for repository_name, dataframe in repositories:
                print "Using ", class_label, " as the class feature."
                print "Working on repository ", repository_name, " with ", len(dataframe.index), "Issues"

                if len(dataframe.index) >= minimum_records:
                    results = execute_analysis(dataframe, class_label + "-" + repository_name, class_label)
                    consolidated_results.extend(results)
                else:
                    print "Not enough issues for analysis: ", len(dataframe.index)

    finally:
        selector.write_results("Binary_Classification.csv", consolidated_results)
        winsound.Beep(2500, 1000)


if __name__ == "__main__":
    main()
