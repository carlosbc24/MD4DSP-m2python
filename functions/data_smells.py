import logging

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
        raise TypeError("Expected number of decimals must be a positive integer")

    # Check precision consistency for all numeric fields
    if field is None:
        # If no specific field is provided, check all numeric fields
        numeric_fields = data_dictionary.select_dtypes(include=['float64', 'Int64', 'int64']).columns
        results = []
        for numeric_field in numeric_fields:
            result = check_precision_consistency(data_dictionary, expected_decimals, numeric_field)
            results.append(result)
        return all(results)

    # If a specific field is provided, check that field
    else:
        if field not in data_dictionary.columns:
            raise ValueError(f"Field '{field}' does not exist in the DataFrame. Skipping precision check.")
        elif not pd.api.types.is_numeric_dtype(data_dictionary[field]):
            # Case 1: The field is not numeric
            print(f"Warning - Field {field} is not numeric. Skipping precision check.")
            return False

        # DataSmell - Precision Inconsistency
        if pd.api.types.is_numeric_dtype(data_dictionary[field]):
            decimals_in_column = data_dictionary[field].dropna().apply(
                lambda x: len(str(float(x)).split(".")[1].rstrip('0')) if '.' in str(float(x)) else 0
            )

            # Count unique decimal lengths in the column
            unique_decimals = decimals_in_column.unique()
            num_unique_decimals = len(unique_decimals)

            if num_unique_decimals > 1:
                # Case 2: Inconsistent decimal places
                print(
                    f"Warning - Column {field} has inconsistent number of decimal places. Found {num_unique_decimals} "
                    f"different decimal lengths.")
                return False
            elif num_unique_decimals == 1 and unique_decimals[0] != expected_decimals:
                # Case 3: Wrong number of decimals
                print(
                    f"Warning - Column {field} has {unique_decimals[0]} decimal places but {expected_decimals} were expected.")
                return False

        return True
