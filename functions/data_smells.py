import pandas as pd

from helpers.logger import print_and_log


def check_precision_consistency(data_dictionary: pd.DataFrame, expected_decimals: int, field: str = None):
    """
    Check if the precision of the data is consistent with the expected precision.
    This function checks if the number of decimal places in a numeric field matches the expected number of decimals.
    If no field is specified, it checks all numeric fields in the DataFrame.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param expected_decimals: (int) Expected number of decimal places
    :param field: (str) Optional field to check; if None, checks all numeric fields
    """
    # Check if the expected_decimals is a non-negative integer
    if not isinstance(expected_decimals, int) or expected_decimals < 0:
        print("Warning - Expected decimals must be a non-negative integer. Skipping precision check.")
        return

    # Check precision consistency for all numeric fields
    if field is None:
        # If no specific field is provided, check all numeric fields
        numeric_fields = data_dictionary.select_dtypes(include=['float64', 'Int64', 'int64']).columns
        for numeric_field in numeric_fields:
            check_precision_consistency(data_dictionary, expected_decimals, numeric_field)

    # If a specific field is provided, check that field
    else:
        if field not in data_dictionary.columns:  # Check if the field exists in the DataFrame
            print(f"Warning - Field {field} does not exist in the data dictionary.")
            return
        elif not pd.api.types.is_numeric_dtype(data_dictionary[field]):  # Check if the field is numeric
            print(f"Warning - Field {field} is not numeric. Skipping precision check.")
            return

        # DataSmell - Precision Inconsistency - Check if the number of decimals in the column matches the expected number
        if pd.api.types.is_float_dtype(data_dictionary[field]):
            decimals_in_column = data_dictionary[field].dropna().apply(
                lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)

            if not decimals_in_column.nunique() == 1 or decimals_in_column.iloc[0] != expected_decimals:
                print_and_log(
                    f"Warning - The number of decimals in column {field} does not match the expected {expected_decimals}.")
