import logging
import unicodedata
import re

import numpy as np
import pandas as pd

from helpers.auxiliar import is_time_string, is_date_string, is_datetime_string, is_float_string, is_integer_string
from helpers.logger import print_and_log
from helpers.enumerations import DataType


def check_precision_consistency(data_dictionary: pd.DataFrame, expected_decimals: int, field: str = None) -> bool:
    """
    Check if the precision of the data is consistent with the expected precision.
    This function checks if the number of decimal places in a numeric field matches the expected number of decimals.
    If no field is specified, it checks all numeric fields in the DataFrame.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param expected_decimals: (int) Expected number of decimal places
    :param field: (str) Optional field to check; if None, checks all numeric fields

    :return: bool indicating if the precision is consistent
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
            print_and_log(f"Warning - Field {field} is not numeric. Skipping precision check.", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Precision Inconsistency in field {field}")
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
                print_and_log(
                    f"Warning - Column {field} has inconsistent number of decimal places. Found {num_unique_decimals} "
                    f"different decimal lengths.", level=logging.WARN)
                print(f"DATA SMELL DETECTED: Precision Inconsistency in field {field}")
                return False
            elif num_unique_decimals == 1 and unique_decimals[0] != expected_decimals:
                # Case 3: Wrong number of decimals
                print_and_log(
                    f"Warning - Column {field} has {unique_decimals[0]} decimal places but {expected_decimals} were "
                    f"expected.", level=logging.WARN)
                print(f"DATA SMELL DETECTED: Precision Inconsistency in field {field}")
                return False

        return True


def check_missing_invalid_value_consistency(data_dictionary: pd.DataFrame, missing_invalid_list: list,
                                            common_missing_invalid_list: list, field: str = None) -> bool:
    """
    Check if there are any missing or invalid values in the DataFrame that are not aligned with the data model definitions.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param missing_invalid_list: (list) List of values defined as missing or invalid in the data model
    :param common_missing_invalid_list: (list) List of common missing or invalid values to compare against
    :param field: (str) Optional field to check; if None, checks all fields

    :return: bool indicating if the field values are consistent with the data model
    """
    if not isinstance(missing_invalid_list, list) or not isinstance(common_missing_invalid_list, list):
        raise TypeError("Both missing_invalid_list and common_missing_invalid_list must be lists")

    # Convert all values list to sets for efficient comparison
    missing_invalid_set = set(missing_invalid_list)
    common_set = set(common_missing_invalid_list)

    def check_field_values(field_name: str):
        """
        Helper function to check values in a single field

        :param field_name: (str) Name of the field to check

        :return: bool indicating if the field values are consistent with the data model
        """
        # Error case: Field does not exist in the DataFrame
        if field_name not in data_dictionary.columns:
            raise ValueError(f"Field '{field_name}' does not exist in the DataFrame. Skipping check.")

        # Convert column values to string and get unique values
        unique_values = set(data_dictionary[field_name].unique())

        # Find values that are in common list but not in model definition
        undefined_values = unique_values.intersection(common_set) - missing_invalid_set

        if undefined_values:
            message = (f"Warning - Possible data smell: The missing or invalid values {list(undefined_values)} in "
                       f"the dataField {field_name} "
                       f"do not align with the definitions in the data model: {list(missing_invalid_set)}")
            print_and_log(message, level=logging.WARN)
            # Case 1: Values in the field are not aligned with the data model definitions
            print(f"DATA SMELL DETECTED: Missing or Invalid Value Inconsistency in field {field_name}")
            return False
        # Case 2: All values in the field are aligned with the data model definitions
        return True

    # Check either all fields or a specific field
    fields_to_check = [field] if field is not None else data_dictionary.columns
    return all(check_field_values(f) for f in fields_to_check)


def check_integer_as_floating_point(data_dictionary: pd.DataFrame, field: str = None) -> bool:
    """
    Checks if any float column in the DataFrame contains only integer values (decimals always .00).
    If so, logs a warning indicating a possible data smell.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all float columns

    :return: (bool) False if a smell is detected, True otherwise.
    """

    def check_column(col_name):
        # First, check if the column is of a float type
        if pd.api.types.is_float_dtype(data_dictionary[col_name]):
            col = data_dictionary[col_name].dropna()
            if not col.empty:
                # Check if all values in the column are integers
                if np.all((col.values == np.floor(col.values))):
                    message = f"Warning - Column '{col_name}' may be an integer disguised as a float."
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Integer as Floating Point in field {col_name}")
                    return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"Field '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        float_fields = data_dictionary.select_dtypes(include=['float', 'float64', 'float32']).columns
        for col in float_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_types_as_string(data_dictionary: pd.DataFrame, field: str, expected_type: DataType) -> bool:
    """
    Check if a column defined as String actually contains only integers, floats, times, dates, or datetimes as string representations.
    If the expected type is not String, check that the values match the expected type.
    Issues a warning if a data smell is detected, or raises an exception if the type does not match the model.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Name of the field (column) to check
    :param expected_type: (DataType) Expected data type as defined in the data model (from helpers.enumerations.DataType)
    :return: (bool) True if the column matches the expected type or no data smell is found, False otherwise
    """

    # Check if the field exists in the DataFrame
    if field not in data_dictionary.columns:
        raise ValueError(f"Field '{field}' does not exist in the DataFrame.")

    col_dtype = data_dictionary[field].dtype

    # If the expected type is String, check if all values are actually another type (integer, float, time, date, datetime)
    if expected_type == DataType.STRING:

        # Convert values to string, remove NaN and strip whitespace
        values = data_dictionary[field].replace('nan', np.nan).dropna().astype(str).str.strip()

        # Detect if the original column is numeric (int or float)
        if pd.api.types.is_integer_dtype(col_dtype) or values.apply(is_integer_string).all():
            print_and_log(f"Warning - Possible data smell: all values in {field} are of type Integer, but the field is defined as String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Integer as String in field {field}")
            return False
        elif pd.api.types.is_float_dtype(col_dtype) or values.apply(is_float_string).all():
            print_and_log(f"Warning - Possible data smell: all values in {field} are of type Float, but the field is defined as String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Float as String in field {field}")
            return False
        elif values.apply(is_time_string).all():
            print_and_log(f"Warning - Possible data smell: all values in {field} are of type Time, but the field is defined as String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Time as String in field {field}")
            return False
        elif values.apply(is_date_string).all():
            print_and_log(f"Warning - Possible data smell: all values in {field} are of type Date, but the field is defined as String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: Date as String in field {field}")
            return False
        elif values.apply(is_datetime_string).all():
            print_and_log(f"Warning - Possible data smell: all values in {field} are of type DateTime, but the field is defined as String in the data model", level=logging.WARN)
            print(f"DATA SMELL DETECTED: DateTime as String in field {field}")
            return False
        # No data smell detected, values are not all of a single other type
        return True
    else:

        # Remove NaN values and convert to the expected type
        values = data_dictionary[field].replace('nan', np.nan).dropna()

        # Type checkers for each expected type
        type_checkers = {
            DataType.INTEGER: lambda v: pd.api.types.is_integer_dtype(data_dictionary[field]) or (
                        pd.api.types.is_numeric_dtype(v) and v.apply(lambda x: float(x).is_integer()).all()),
            DataType.FLOAT: lambda v: pd.api.types.is_float_dtype(
                data_dictionary[field]) or pd.api.types.is_numeric_dtype(v),
            DataType.DOUBLE: lambda v: pd.api.types.is_float_dtype(
                data_dictionary[field]) or pd.api.types.is_numeric_dtype(v),
            DataType.TIME: lambda v: pd.api.types.is_datetime64_dtype(v) or v.apply(
                lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).all(),
            DataType.DATE: lambda v: pd.api.types.is_datetime64_dtype(v) or v.apply(
                lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).all(),
            DataType.DATETIME: lambda v: pd.api.types.is_datetime64_dtype(v) or v.apply(
                lambda x: isinstance(x, (pd.Timestamp, np.datetime64))).all(),
            DataType.BOOLEAN: lambda v: pd.api.types.is_bool_dtype(v) or v.apply(
                lambda x: isinstance(x, (bool, np.bool_))).all(),
            DataType.STRING: lambda v: pd.api.types.is_string_dtype(v) or v.apply(lambda x: isinstance(x, str)).all()
        }

        checker = type_checkers.get(expected_type)
        if checker is None:
            raise ValueError(f"Unknown expected_type '{expected_type}' for field '{field}'")
        if not checker(values):
            raise TypeError(f"Expected data for column {field} is {expected_type.name}, but got {col_dtype.name}")
        return True


def check_special_character_spacing(data_dictionary: pd.DataFrame, field: str = None) -> bool:
    """
    Checks if string columns contain accents, uppercase letters, extra spaces, or special characters
    that do not align with the recommended data format for string operations.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param field: (str) Optional field to check; if None, checks all string columns
    :return: (bool) False if a smell is detected, True otherwise.
    """

    def clean_text(text):
        """Helper function to clean text by removing accents, special characters, extra spaces and converting to lowercase"""
        if pd.isna(text) or text == '':
            return text
        # Convert to string in case it's not
        text = str(text)
        # Remove accents, special characters, normalize spaces and convert to lowercase
        return re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z0-9\s]', '', ''.join([c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'])).lower()).strip()

    def check_column(col_name):
        # Only check string columns
        if pd.api.types.is_string_dtype(data_dictionary[col_name]) or data_dictionary[col_name].dtype == 'object':
            col = data_dictionary[col_name].dropna()
            if not col.empty:
                # Apply cleaning function to all values
                cleaned_values = col.apply(clean_text)

                # Check if any value changed after cleaning (indicating presence of special chars, spaces, etc.)
                if not (col == cleaned_values).all():
                    message = f"Warning - Possible data smell: the values in {col_name} contain accents, uppercase letters, extra spaces, or special characters that do not align with the recommended data format for string operations."
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Special Character/Spacing in field {col_name}")
                    return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"Field '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all string/object columns
        string_fields = data_dictionary.select_dtypes(include=['object', 'string']).columns
        for col in string_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


def check_suspect_distribution(data_dictionary: pd.DataFrame, min_value: float, max_value: float, field: str = None) -> bool:
    """
    Checks if continuous data fields have values outside the range defined in the data model.
    If so, logs a warning indicating a possible data smell.

    :param data_dictionary: (pd.DataFrame) DataFrame containing the data
    :param min_value: (float) Minimum value allowed according to the data model
    :param max_value: (float) Maximum value allowed according to the data model
    :param field: (str) Optional field to check; if None, checks all numeric fields
    :return: (bool) False if a smell is detected, True otherwise.
    """
    
    # Validate input parameters
    if not isinstance(min_value, (int, float)) or not isinstance(max_value, (int, float)):
        raise TypeError("min_value and max_value must be numeric")
    
    if min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")

    def check_column(col_name):
        # Only check numeric columns (continuous data)
        if pd.api.types.is_numeric_dtype(data_dictionary[col_name]):
            col = data_dictionary[col_name].dropna()
            if not col.empty:
                # Check if any values are outside the defined range
                out_of_range = (col < min_value) | (col > max_value)
                if out_of_range.any():
                    message = f"Warning - Possible data smell: The range of values of dataField {col_name} do not align with the definitions in the data-model"
                    print_and_log(message, level=logging.WARN)
                    print(f"DATA SMELL DETECTED: Suspect Distribution in field {col_name}")
                    return False
        return True

    if field is not None:
        if field not in data_dictionary.columns:
            raise ValueError(f"Field '{field}' does not exist in the DataFrame.")
        return check_column(field)
    else:
        # If DataFrame is empty, return True (no smell)
        if data_dictionary.empty:
            return True
        # Check all numeric columns
        numeric_fields = data_dictionary.select_dtypes(include=['number', 'float64', 'float32', 'int64', 'int32']).columns
        for col in numeric_fields:
            result = check_column(col)
            if not result:
                return result  # Return on the first smell found
    return True


