"""
This modules identifies inflated issues without supervised learning support.
"""

import pandas as pd

import preprocessing

ALL_ISSUES_CSV = "C:\Users\Carlos G. Gavidia\git\github-data-miner\UNFILTERED\Release_Counter_UNFILTERED.csv"


def main():
    """
    Initial execution point.
    :return: None.
    """
    unfitltered_dataframe = pd.read_csv(ALL_ISSUES_CSV)
    print "Initial CSV load ", unfitltered_dataframe.shape

    changer_column = 'Priority Changer'
    with_change_dataframe = unfitltered_dataframe.loc[unfitltered_dataframe[changer_column].notnull()]
    print "Issues with Priority Changes ", with_change_dataframe.shape

    not_reporter_dataframe = with_change_dataframe[
        with_change_dataframe[changer_column] != with_change_dataframe['Reported By']]
    print "Issues where the changer is not the reporter ", not_reporter_dataframe.shape

    original_priority_column = 'Encoded Original Priority'
    new_priority_column = 'Encoded New Priority'

    original_priority_desc = 'Original Priority'
    new_priority_desc = 'New Priority'

    not_reporter_dataframe[original_priority_column] = not_reporter_dataframe[original_priority_desc].map(
        preprocessing.PRIORITY_MAP)
    not_reporter_dataframe[new_priority_column] = not_reporter_dataframe[new_priority_desc].map(
        preprocessing.PRIORITY_MAP)

    inflated_dataframe = not_reporter_dataframe[
        not_reporter_dataframe[original_priority_column] < not_reporter_dataframe[new_priority_column]]
    print "Inflated issues ", inflated_dataframe.shape
    print "Priority distribution on Inflated Issues: Original Value ", inflated_dataframe[
        original_priority_desc].value_counts()
    print "Priority distribution on Inflated Issues: New Value ", inflated_dataframe[new_priority_desc].value_counts()
    print "Project distribution on Inflated Issues ", inflated_dataframe['']

    deflated_dataframe = not_reporter_dataframe[
        not_reporter_dataframe[original_priority_column] > not_reporter_dataframe[new_priority_column]]
    print "Deflated issues ", deflated_dataframe.shape


if __name__ == "__main__":
    main()
