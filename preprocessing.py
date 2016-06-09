"""
This modules handles the CSV preprocessing before executing the ML code.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_FILE = "C:\Users\Carlos G. Gavidia\git\github-data-miner\Release_Counter_.csv"


def load_original_dataframe():
    """
    Loads the issues CSV without additional filtering. Note that all this issues have at least one commit in Git,
    have been in a valid resolved state and were reported and resolved by different persons.

    Also, we will filter issues that don't have priority information.

    :return: CSV data as a data frame.
    """
    original_dataframe = pd.read_csv(CSV_FILE)

    print "Loaded ", len(
        original_dataframe.index), " resolved issues with Git Commits, solved by a third-party from ", CSV_FILE

    original_dataframe = original_dataframe.dropna(subset=['Priority'])
    print len(original_dataframe.index), "after excluding empty priorities."

    return original_dataframe


def filter_issues_dataframe(original_dataframe, repository=None, priority_changer=True, git_metrics=True):
    """
    Returns a dataframe of issues that: Are resolved, have a commit in Git, were not resolved by
    the reported and had a priority change not made by the reporter.

    Also, that does not have the distance in releases null.

    :return: Filtered dataframe
    """

    print "Total columns in dataframe: ", len(original_dataframe.columns)

    priority_changer_columnn = 'Priority Changer'
    issue_dataframe = original_dataframe

    if priority_changer:
        issue_dataframe = issue_dataframe.dropna(subset=[priority_changer_columnn])
        issue_dataframe = issue_dataframe[issue_dataframe[priority_changer_columnn] != issue_dataframe['Reported By']]
        print len(issue_dataframe.index), " issues had a priority corrected by a third-party."

    if git_metrics:
        issue_dataframe = issue_dataframe.dropna(subset=['GitHub Distance in Releases', 'Git Resolution Time'])
        print len(issue_dataframe.index), "have release information in Git."

    if repository:
        repository_column = 'Git Repository'
        issue_dataframe = issue_dataframe[issue_dataframe[repository_column] == repository]
        issue_dataframe = issue_dataframe.drop(repository_column, 1)

        print len(issue_dataframe.index), " issues corresponding to repository ", repository

    return issue_dataframe


def encode_class_label(issue_dataframe, class_label):
    """
    Replaces the priority as String for a numerical value. It also adds the Severe column (Boolean, true if it is a Blocker or Critical issue.)
    :param issue_dataframe: Original dataframe
    :param class_label: Column containing encoded Priority Information
    :return: New dataframe
    """
    priority_mapping = {'Blocker': 1,
                        'Critical': 2,
                        'Major': 3,
                        'Minor': 4,
                        'Trivial': 5}

    original_label = 'Priority'
    issue_dataframe[class_label] = issue_dataframe[original_label].map(priority_mapping)
    issue_dataframe['Severe'] = issue_dataframe[class_label] <= 2

    simplified_mapping = {'Blocker': 1,
                          'Critical': 1,
                          'Major': 2,
                          'Minor': 3,
                          'Trivial': 3}
    issue_dataframe['Simplified ' + class_label] = issue_dataframe[original_label].map(simplified_mapping)

    return issue_dataframe


def encode_and_split(issues_dataframe, class_label, numerical_features, nominal_features):
    """
    Filters only the relevant features, encodes de class label and the encodes nominal features.

    :param issues_dataframe: Original dataframe.
    :param class_label: Class label.
    :param numerical_features: Numerical features.
    :param nominal_features: Nominal features.
    :return: Dataframe with features, series with labels.
    """
    issue_dataframe = encode_class_label(issues_dataframe, class_label)

    # class_label = 'Severe'
    # class_label = 'Simplified ' + class_label

    issue_dataframe = issue_dataframe[numerical_features + nominal_features + [class_label]]
    issue_dataframe = encode_nominal_features(issue_dataframe, nominal_features)

    encoded_priorities = issue_dataframe[class_label].reset_index()
    issue_dataframe = issue_dataframe.drop([class_label], axis=1)

    print "Number of features: ", len(issue_dataframe.columns)

    return issue_dataframe, encoded_priorities[class_label]


def encode_nominal_features(issue_dataframe, nominal_features):
    """
    Performs a one-hot encoding on the nominal features of the dataset.
    :param issue_dataframe: Original Dataframe
    :param nominal_features: List of nominal features.
    :return: New dataframe.
    """

    if nominal_features:
        dummy_dataframe = pd.get_dummies(issue_dataframe[nominal_features])
        temp_dataframe = pd.concat([issue_dataframe, dummy_dataframe], axis=1)
        temp_dataframe = temp_dataframe.drop(nominal_features, axis=1)

        return temp_dataframe

    else:
        return issue_dataframe


def escale_numerical_features(numerical_features, issues_train, issues_test=None):
    """
    Standarizes numerical information.
    :param issues_train: Train issues for the scaler.
    :param issues_test: Issue set to also be scaled.
    :param numerical_features: Features to scale

    :return: Train and test sets standarized.
    """

    issues_train_std = issues_train.copy()

    if issues_test is not None:
        issues_test_std = issues_test.copy()
    else:
        issues_test_std = None

    for feature in numerical_features:
        scaler = StandardScaler()

        issues_train_std[feature] = scaler.fit_transform(issues_train[feature].reshape(-1, 1))

        if issues_test is not None:
            issues_test_std[feature] = scaler.transform(issues_test[feature].reshape(-1, 1))

    return issues_train_std, issues_test_std
