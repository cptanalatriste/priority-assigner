"""
This modules builds a priority assigner considering possible inflated issues in the training set
"""

import assigner
import selector


def main():
    """
    Initial execution point.
    :return: None.
    """

    original_dataframe = assigner.load_original_dataframe()
    original_dataframe = assigner.filter_issues_dataframe(original_dataframe, priority_changer=False)

    # Excluding the project membership feature.
    issues, priorities = assigner.prepare_for_training(original_dataframe, assigner.CLASS_LABEL,
                                                       assigner.NUMERICAL_FEATURES, [])

    issues_train_std, priority_train, issues_test_std, priority_test = selector.split_train_test(issues=issues,
                                                                                                 priorities=priorities)

    # TODO All these previous steps should be moved to their own module.

    print "issues_train_std.head() ", issues_train_std.head()
    print "priority_train.head() ", priority_train.head()


if __name__ == "__main__":
    main()
