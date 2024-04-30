# Importing libraries
from datetime import datetime
# Importing functions and classes from packages
from typing import Union
from datetime import datetime
import numpy as np
import pandas as pd
from helpers.auxiliar import compare_numbers, count_abs_frequency
from helpers.enumerations import Belong, Operator, Closure
from helpers.transform_aux import getOutliers


class ContractsPrePost:

    def checkFieldRange(self, fields: list, data_dictionary: pd.DataFrame, belongOp: Belong) -> bool:
        """
        Check if fields meets the condition of belongOp in data_dictionary.
        If belongOp is Belong.BELONG, then it checks if all fields are in data_dictionary.
        If belongOp is Belong.NOTBELONG, then it checks if any field in 'fields' are not in data_dictionary.

        :param fields: list of columns
        :param data_dictionary: data dictionary
        :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG

        :return: if fields meets the condition of belongOp in data_dictionary
        :rtype: bool
        """
        if belongOp == Belong.BELONG:
            for field in fields:
                if field not in data_dictionary.columns:
                    return False # Case 1
            return True # Case 2
        elif belongOp == Belong.NOTBELONG:
            for field in fields:
                if field not in data_dictionary.columns:
                    return True # Case 3
            return False # Case 4

    def checkFixValueRange(self, value: Union[str, float, datetime], data_dictionary: pd.DataFrame, belongOp: Belong,
                           field: str = None,
                           quant_abs: int = None, quant_rel: float = None, quant_op: Operator = None) -> bool:
        """
        Check if fields meets the condition of belongOp in data_dictionary

        :param value: float value to check
        :param data_dictionary: data dictionary
        :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param field: dataset column in which value will be checked
        :param quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
               Operator.LESS=3, Operator.EQUAL=4
        :param quant_abs: integer which represents the absolute number of times that value should appear
                            with respect the enum operator quant_op
        :param quant_rel: float which represents the relative number of times that value should appear
                            with respect the enum operator quant_op

        :return: if fields meets the condition of belongOp in data_dictionary and field
        :rtype: bool
        """
        data_dictionary = data_dictionary.replace({
            np.nan: None})  # Replace NaN values with None to avoid error when comparing None with NaN. As the dataframe is of floats, the None are converted to NaN
        if value is not None and type(value) is not str and type(
                value) is not pd.Timestamp: # Before casting, it is checked that value is not None, str or datetime(Timestamp), so that only the int are casted
            # Before casting, it is checked that value is not None, str or datetime(Timestamp), so that only the int are casted
            value = float(
                value)  # Cast the float to avoid errors when comparing the value with the values of the dataframe

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if value is in data_dictionary
                    return True if value in data_dictionary.values else False # Case 1 y 2
                else:
                    if quant_rel is not None and quant_abs is None:  # Check if value is in data_dictionary and if it meets the condition of quant_rel
                        return True if value in data_dictionary.values and compare_numbers( # Case 3 y 4
                            count_abs_frequency(value, data_dictionary) / data_dictionary.size,
                            quant_rel,
                            quant_op) else False  # If field is None, in place of looking in a column, it looks in the whole dataframe
                        # Important to highlight that it is necessary to use dropna=False to count the NaN values in case value is None
                    elif quant_rel is not None and quant_abs is not None:
                        # If both are provided, a ValueError is raised
                        raise ValueError( # Case 4.5
                            "quant_rel and quant_abs can't have different values than None at the same time")
                    elif quant_abs is not None:
                        return True if value in data_dictionary.values and compare_numbers( # Case 5 y 6
                            count_abs_frequency(value, data_dictionary),
                            quant_abs,
                            quant_op) else False  # If field is None, in place of looking in a column, it looks in the whole dataframe
                    else:
                        raise ValueError(  # Case 7
                            "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    return True if value not in data_dictionary.values else False  # Case 8 y 9
                else:
                    raise ValueError(
                        "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Case 10
        else:
            if field is not None:
                if field not in data_dictionary.columns:  # It checks that the column exists in the dataframe
                    raise ValueError(f"Column '{field}' not found in data_dictionary.")  # Case 10.5
                if belongOp == Belong.BELONG: # If the value should belong to the column
                    if quant_op is None:
                        return True if value in data_dictionary[field].values else False  # Case 11 y 12
                    else:
                        if quant_rel is not None and quant_abs is None:
                            return True if value in data_dictionary[field].values and compare_numbers(  # Case 13 y 14
                                data_dictionary[field].value_counts(dropna=False).get(value, 0)
                                / data_dictionary.size, quant_rel,
                                quant_op) else False  # It is important to highlight that it is necessary to use dropna=False to count the NaN values in case value is None
                        elif quant_rel is not None and quant_abs is not None:
                            # If both are provided, a ValueError is raised
                            raise ValueError(  # Case 14.5
                                "quant_rel and quant_abs can't have different values than None at the same time")
                        elif quant_abs is not None:
                            return True if value in data_dictionary[field].values and compare_numbers(
                                data_dictionary[field].value_counts(dropna=False).get(value, 0),
                                quant_abs, quant_op) else False  # Case 15 y 16
                        else:  # quant_rel is None and quant_abs is None
                            raise ValueError(  # Case 17
                                "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        return True if value not in data_dictionary[field].values else False  # Case 18 y 19
                    else:  # Case 20
                        raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")

    def checkIntervalRangeFloat(self, left_margin: float, right_margin: float, data_dictionary: pd.DataFrame,
                                closureType: Closure, belongOp: Belong, field: str = None) -> bool:
        """
            Check if the data_dictionary meets the condition of belongOp in the interval defined by leftMargin and rightMargin with the closureType.
            If field is None, it does the check in the whole data_dictionary. If not, it does the check in the column specified by field.

            :param left_margin: float value which represents the left margin of the interval
            :param right_margin: float value which represents the right margin of the interval
            :param data_dictionary: data dictionary
            :param closureType: enum operator which can be Closure.openOpen=0, Closure.openClosed=1,
                                Closure.closedOpen=2, Closure.closedClosed=3
            :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
            :param field: dataset column in which value will be checked

            :return: if data_dictionary meets the condition of belongOp in the interval defined by leftMargin and rightMargin with the closureType
        """
        if left_margin > right_margin:
            raise ValueError("Error: leftMargin should be less than or equal to rightMargin")  # Case 0

        def check_condition(min_val: float, max_val: float) -> bool:
            if closureType == Closure.openOpen:
                return True if (min_val > left_margin) & (max_val < right_margin) else False
            elif closureType == Closure.openClosed:
                return True if (min_val > left_margin) & (max_val <= right_margin) else False
            elif closureType == Closure.closedOpen:
                return True if (min_val >= left_margin) & (max_val < right_margin) else False
            elif closureType == Closure.closedClosed:
                return True if (min_val >= left_margin) & (max_val <= right_margin) else False
            else:
                raise ValueError("Error: closureType should be openOpen, openClosed, closedOpen, or closedClosed")

        if field is None:
            data_dictionary = data_dictionary.select_dtypes(
                include=['int', 'float'])  # It discards the columns that are not of type int or float
            return check_condition(data_dictionary.min().min(),
                                   data_dictionary.max().max()) if belongOp == Belong.BELONG else not check_condition(
                data_dictionary.min().min(), data_dictionary.max().max())  # Cases 1-16
        else:
            if field not in data_dictionary.columns:  # It checks that the column exists in the dataframe
                raise ValueError(f"Column '{field}' not found in data_dictionary.")  # Case 16.5
            if np.issubdtype(data_dictionary[field].dtype, np.number):
                return check_condition(data_dictionary[field].min(),
                                       data_dictionary[
                                           field].max()) if belongOp == Belong.BELONG else not check_condition(
                    data_dictionary[field].min(), data_dictionary[field].max())  # Cases 17-32
            else:
                raise ValueError("Error: field should be a float")  # Case 33

    def checkMissingRange(self, belongOp: Belong, data_dictionary: pd.DataFrame, field: str = None,
                          missing_values: list = None, quant_abs: int = None, quant_rel: float = None,
                          quant_op: Operator = None) -> bool:
        """
        Check if the data_dictionary meets the condition of belongOp with respect to the missing values defined in missing_values.
        If field is None, it does the check in the whole data_dictionary. If not, it does the check in the column specified by field.

        :param missing_values: list of missing values
        :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param data_dictionary: data dictionary
        :param field: dataset column in which value will be checked

        :return: if data_dictionary meets the condition of belongOp with respect to the missing values defined in missing_values
        """
        if missing_values is not None:
            for i in range(len(missing_values)):
                if isinstance(missing_values[i], int):
                    missing_values[i] = float(missing_values[i])

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Checks if there are any missing values from the list 'missing_values' in data_dictionary
                    if data_dictionary.isnull().values.any():
                        return True  # Case 1
                    else:  # If there aren't null python values in data_dictionary, it checks if there are any of the
                        # missing values in the list 'missing_values'
                        if missing_values is not None:
                            return True if any(
                                value in missing_values for value in
                                data_dictionary.values.flatten()) else False  # Case 2 y 3
                        else:  # If the list is None, it returns False. It checks that in fact there aren't any missing values
                            return False  # Case 4
                else:
                    if quant_rel is not None and quant_abs is None:  # Check there are any null python values or missing values from the list 'missing_values' in data_dictionary and if it meets the condition of quant_rel and quant_op
                        if (data_dictionary.isnull().values.any() or (missing_values is not None and any(
                                value in missing_values for value in
                                data_dictionary.values.flatten()))) and compare_numbers(
                            (data_dictionary.isnull().values.sum() + sum(
                                [count_abs_frequency(value, data_dictionary) for value in
                                 (missing_values if missing_values is not None else [])])) / data_dictionary.size,
                            quant_rel, quant_op):
                            return True  # Case 5
                        else:
                            return False  # Case 6
                    elif quant_rel is not None and quant_abs is not None:
                        # If both are provided, a ValueError is raised
                        raise ValueError(
                            "quant_rel and quant_abs can't have different values than None at the same time")  # Case 7
                    elif quant_abs is not None:  # Check there are any null python values or missing values from the
                        # list 'missing_values' in data_dictionary and if it meets the condition of quant_abs and
                        # quant_op
                        if (data_dictionary.isnull().values.any() or (
                                missing_values is not None and any(
                            value in missing_values for value in data_dictionary.values.flatten()))) and compare_numbers(
                            data_dictionary.isnull().values.sum() + sum(
                                [count_abs_frequency(value, data_dictionary) for value in
                                 (missing_values if missing_values is not None else [])]),
                            quant_abs, quant_op):
                            return True  # Case 8
                        else:
                            return False  # Case 9
                    else:
                        raise ValueError(
                            "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is "
                            "not None")  # Case 10
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    # Check that there aren't any null python values or missing values from the list 'missing_values'
                    # in data_dictionary
                    if missing_values is not None:
                        return True if not data_dictionary.isnull().values.any() and not any(
                            value in missing_values for value in
                            data_dictionary.values.flatten()) else False  # Case 11 y 12
                    else:  # If the list is None, it checks that there aren't any python null values in data_dictionary
                        return True if not data_dictionary.isnull().values.any() else False  # Case 13 y 13.5
                else:
                    raise ValueError(
                        "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Case 14
        else:
            if field is not None:  # Check to make code more legible
                if field not in data_dictionary.columns:  # Checks that the column exists in the dataframe
                    raise ValueError(f"Column '{field}' not found in data_dictionary.")  # Case 15
                if belongOp == Belong.BELONG:
                    if quant_op is None:  # Check that there are null python values or missing values from the list
                        # 'missing_values' in the column specified by field
                        if data_dictionary[field].isnull().values.any():
                            return True  # Case 16
                        else:  # If there aren't null python values in data_dictionary, it checks if there are any of the
                            # missing values in the list 'missing_values'
                            if missing_values is not None:
                                return True if any(
                                    value in missing_values for value in
                                    data_dictionary[field].values) else False  # Case 17 y 18
                            else:  # If the list is None, it returns False. It checks that in fact there aren't any missing values
                                return False  # Case 19
                    else:
                        if quant_rel is not None and quant_abs is None:  # Check there are null python values or
                            # missing values from the list 'missing_values' in the column specified by field and if it
                            # meets the condition of quant_rel and quant_op
                            if (data_dictionary[field].isnull().values.any() or (
                                    missing_values is not None and any(value in missing_values for value in
                                                                       data_dictionary[
                                                                           field].values))) and compare_numbers(
                                (data_dictionary[field].isnull().values.sum() + sum(
                                    [count_abs_frequency(value, data_dictionary, field) for value in
                                     (missing_values if missing_values is not None else [])])) / data_dictionary[
                                    field].size, quant_rel, quant_op):
                                return True  # Case 20
                            else:
                                return False  # Case 21
                            # Relative frequency respect to the data specified by field
                        elif quant_rel is not None and quant_abs is not None:
                            # If both are provided, a ValueError is raised
                            raise ValueError(
                                "quant_rel and quant_abs can't have different values than None at the same time")  # Case 22
                        elif quant_abs is not None:  # Check there are null python values or missing values from the
                            # list 'missing_values' in the column specified by field and if it meets the condition of
                            # quant_abs and quant_op
                            if (data_dictionary[field].isnull().values.any() or (
                                    missing_values is not None and any(value in missing_values for value in
                                                                       data_dictionary[
                                                                           field].values))) and compare_numbers(
                                data_dictionary[field].isnull().values.sum() + sum(
                                    [count_abs_frequency(value, data_dictionary, field) for value in
                                     (missing_values if missing_values is not None else [])]),
                                quant_abs, quant_op):
                                return True  # Case 23
                            else:
                                return False  # Case 24
                        else:  # quant_rel is None and quant_abs is None
                            raise ValueError(
                                "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")  # Case 25
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        # Check that there aren't any null python values or missing values from the list
                        # 'missing_values' in the column specified by field
                        if missing_values is not None:  # Check that there are missing values in the list 'missing_values'
                            return True if not data_dictionary[field].isnull().values.any() and not any(
                                value in missing_values for value in
                                data_dictionary[field].values) else False  # Case 26 y 27
                        else:  # If the list is None, it checks that there aren't any python null values in the column specified by field
                            return True if not data_dictionary[field].isnull().values.any() else False  # Case 28 y 29
                    else:
                        raise ValueError(
                            "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Case 30

    def checkInvalidValues(self, belongOp: Belong, data_dictionary: pd.DataFrame, invalid_values: list,
                           field: str = None, quant_abs: int = None, quant_rel: float = None,
                           quant_op: Operator = None) -> bool:
        """
        Check if the data_dictionary meets the condition of belongOp with respect to the invalid values defined in invalid_values.
        If field is None, it does the check in the whole data_dictionary. If not, it does the check in the column specified by field.

        :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param data_dictionary: data dictionary
        :param invalid_values: list of invalid values
        :param field: dataset column in which value will be checked
        :param quant_abs: integer which represents the absolute number of times that value should appear
                            with respect the enum operator quant_op
        :param quant_rel: float which represents the relative number of times that value should appear
                            with respect the enum operator quant_op
        :param quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
               Operator.LESS=3, Operator.EQUAL=4

        :return: if data_dictionary meets the condition of belongOp with respect to the invalid values defined in invalid_values
        """
        if invalid_values is not None:
            for i in range(len(invalid_values)):
                if isinstance(invalid_values[i], int):
                    invalid_values[i] = float(invalid_values[i])

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if there are any invalid values in data_dictionary
                    if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                        if any(value in invalid_values for value in data_dictionary.values.flatten()):
                            return True  # Case 1
                        else:
                            return False  # Case 2
                    else:  # If the list is None, it returns False. It checks that in fact there aren't any invalid values
                        return False  # Case 3
                else:
                    if quant_rel is not None and quant_abs is None:  # Check there are any invalid values in
                        # data_dictionary and if it meets the condition of quant_rel and quant_op (relative frequency)
                        if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                            if any(value in invalid_values for value in
                                   data_dictionary.values.flatten()) and compare_numbers(
                                sum([count_abs_frequency(value, data_dictionary) for value in
                                     invalid_values]) / data_dictionary.size, quant_rel, quant_op):
                                return True  # Case 4
                            else:
                                return False  # Case 5
                        else:  # If the list is None, it returns False
                            return False  # Case 6
                    elif quant_abs is not None and quant_rel is None:  # Check there are any invalid values in
                        # data_dictionary and if it meets the condition of quant_abs and quant_op (absolute frequency)
                        if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                            if any(value in invalid_values for value in
                                   data_dictionary.values.flatten()) and compare_numbers(
                                sum([count_abs_frequency(value, data_dictionary) for value in
                                     invalid_values]), quant_abs, quant_op):
                                return True  # Case 7
                            else:
                                return False  # Case 8
                        else:  # If the list is None, it returns False
                            return False  # Case 9
                    elif quant_abs is not None and quant_rel is not None:
                        # If both are provided, a ValueError is raised
                        raise ValueError(
                            "quant_rel and quant_abs can't have different values than None at the same time")  # Case 10
                    else:
                        raise ValueError(
                            "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is "
                            "not None")  # Case 11
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    # Check that there aren't any invalid values in data_dictionary
                    return True if not (invalid_values is not None and any(
                        value in invalid_values for value in
                        data_dictionary.values.flatten())) else False  # Case 12 y 13
                else:
                    raise ValueError(
                        "Error: quant_op, quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Case 14
        else:
            if field is not None:
                if field not in data_dictionary.columns:
                    raise ValueError(f"Column '{field}' not found in data_dictionary.")  # Case 15
                if belongOp == Belong.BELONG:
                    if quant_op is None:  # Check that there are invalid values in the column specified by field
                        if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                            if any(value in invalid_values for value in data_dictionary[field].values):
                                return True  # Case 16
                            else:
                                return False  # Case 17
                        else:  # If the list is None, it returns False
                            return False  # Case 18
                    else:
                        if quant_rel is not None and quant_abs is None:  # Check there are invalid values in the
                            # column specified by field and if it meets the condition of quant_rel and quant_op
                            # (relative frequency)
                            if invalid_values is not None:  # Checks that there are invalid values in the list 'invalid_values'
                                if any(value in invalid_values for value in
                                       data_dictionary[field].values) and compare_numbers(
                                    sum([count_abs_frequency(value, data_dictionary, field) for value in
                                         invalid_values]) / data_dictionary[field].size, quant_rel, quant_op):
                                    return True  # Case 19
                                else:
                                    return False  # Case 20
                            else:  # If the list is None, it returns False
                                return False  # Case 21
                        elif quant_abs is not None and quant_rel is None:  # Check there are invalid values in the
                            # column specified by field and if it meets the condition of quant_abs and quant_op
                            # (absolute frequency)
                            if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                                if any(value in invalid_values for value in
                                       data_dictionary[field].values) and compare_numbers(
                                    sum([count_abs_frequency(value, data_dictionary, field) for value in
                                         invalid_values]), quant_abs, quant_op):
                                    return True  # Case 22
                                else:
                                    return False  # Case 23
                            else:  # If the list is None, it returns False
                                return False  # Case 24
                        elif quant_abs is not None and quant_rel is not None:
                            # If both are provided, a ValueError is raised
                            raise ValueError(
                                "quant_rel and quant_abs can't have different values than None at the same time")  # Case 25
                        else:
                            raise ValueError(
                                "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")  # Case 26
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        # Check that there aren't any invalid values in the column specified by field
                        if invalid_values is not None: # Checks that there are invalid values in the list 'invalid_values'
                            return True if not any(
                                value in invalid_values for value in
                                data_dictionary[field].values) else False  # Case 27 y 28
                        else:  # If the list is None, it returns True
                            return True  # Case 29
                    else:
                        raise ValueError(
                            "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Case 30

    def checkOutliers(self, data_dictionary: pd.DataFrame, belongOp: Belong = None, field: str = None,
                      quant_abs: int = None, quant_rel: float = None, quant_op: Operator = None) -> bool:
        """
        Check if there are outliers in the numeric columns of data_dictionary. The Outliers are calculated using the IQR method, so the outliers are the values that are
        below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR

        :param data_dictionary: dataframe with the data
        :param axis_param: axis to get the outliers. If axis_param is None, the outliers are calculated for the whole dataframe.
        If axis_param is 0, the outliers are calculated for each column. If axis_param is 1, the outliers are calculated for each row (although it is not recommended as it is not a common use case)

        :return: boolean indicating if there are outliers in the data_dictionary
        """
        data_dictionary_copy = data_dictionary.copy()
        outlier = 1  # 1 is the value that is going to be used to check if there are outliers in the dataframe

        if field is None:
            data_dictionary_copy = getOutliers(data_dictionary=data_dictionary_copy, field=None, axis_param=None)
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if there are any invalid values in data_dictionary
                    if outlier in data_dictionary_copy.values:
                        return True  # Case 1
                    else:
                        return False  # Case 2
                else:
                    if quant_rel is not None and quant_abs is None:  # Check there are any invalid values in
                        # data_dictionary and if it meets the condition of quant_rel and quant_op (relative frequency)
                        if compare_numbers(count_abs_frequency(outlier, data_dictionary_copy) / data_dictionary_copy.size,
                                           quant_rel, quant_op):
                            return True  # Case 3
                        else:
                            return False  # Case 4
                    elif quant_abs is not None and quant_rel is None:  # Check there are any invalid values in
                        # data_dictionary and if it meets the condition of quant_abs and quant_op (absolute frequency)
                        if compare_numbers(count_abs_frequency(outlier, data_dictionary_copy), quant_abs, quant_op):
                            return True  # Case 5
                        else:
                            return False  # Case 6
                    elif quant_abs is not None and quant_rel is not None:
                            # If both are provided, a ValueError is raised
                        raise ValueError(
                            "quant_rel and quant_abs can't have different values than None at the same time")  # Case 7
                    else:
                        raise ValueError(
                            "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is "
                            "not None")  # Case 8
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    return True if not (outlier in data_dictionary_copy.values) else False  # Case 9 y 10
                else:
                    raise ValueError("Error: quant_op, quant_rel and quant_abs should be None when belongOp is "
                                     "NOTBELONG")  # Case 11
        else:
            if field is not None:
                if field not in data_dictionary.columns:
                    raise ValueError(f"Column '{field}' not found in data_dictionary.")  # Case 12

                data_dictionary_copy = getOutliers(data_dictionary=data_dictionary_copy, field=field, axis_param=None)
                if belongOp == Belong.BELONG:
                    if quant_op is None:  # Check that there are invalid values in the column specified by field
                        if outlier in data_dictionary_copy[field].values:
                            return True  # Case 13
                        else:
                            return False  # Case 14
                    else:
                        if quant_rel is not None and quant_abs is None:  # Check there are invalid values in the
                            # column specified by field and if it meets the condition of quant_rel and quant_op
                            # (relative frequency)
                            if compare_numbers(
                                    count_abs_frequency(outlier, data_dictionary_copy) / data_dictionary_copy[field].size,
                                    quant_rel, quant_op):
                                return True  # Case 15
                            else:
                                return False  # Case 16
                        elif quant_abs is not None and quant_rel is None:  # Check there are invalid values in the
                            # column specified by field and if it meets the condition of quant_abs and quant_op
                            # (absolute frequency)
                            if compare_numbers(count_abs_frequency(outlier, data_dictionary_copy), quant_abs, quant_op):
                                return True  # Case 17
                            else:
                                return False  # Case 18
                        elif quant_abs is not None and quant_rel is not None:
                            # If both are provided, a ValueError is raised
                            raise ValueError(
                                "quant_rel and quant_abs can't have different values than None at the same time")  # Case 19
                        else:
                            raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")  # Case 20
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        # Check that there aren't any invalid values in the column specified by field
                        return True if not (outlier in data_dictionary_copy[field].values) else False  # Case 21 y 22
                    else:
                        raise ValueError(
                            "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Case 23
