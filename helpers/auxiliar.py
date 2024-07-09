# Importing enumerations from packages
from typing import Union

# Importing libraries
import numpy as np
import pandas as pd

from helpers.enumerations import Operator, DataType, Closure


def format_duration(seconds: float) -> str:
    """
    Format duration from seconds to hours, minutes, seconds and milliseconds

    :param seconds: (float) Duration in seconds
    :return: formated_duration: (str) Duration in hours, minutes, seconds and milliseconds
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    miliseconds = seconds - int(seconds)
    formated_duration = (f"{hours} hours, {minutes} minutes, {int(seconds)} seconds and {int(miliseconds * 1000)} "
                         f"milliseconds")
    return formated_duration


def compare_numbers(rel_abs_number: Union[int, float], quant_rel_abs: Union[int, float], quant_op: Operator) -> bool:
    """
    Compare two numbers with the operator quant_op

    :param rel_abs_number: (Union[int, float]) relative or absolute number to compare with the previous one
    :param quant_rel_abs: (Union[int, float]) relative or absolute number to compare with the previous one
    :param quant_op: (Operator) operator to compare the two numbers

    :return: if rel_abs_number meets the condition of quant_op with quant_rel_abs
    """
    if quant_op == Operator.GREATEREQUAL:
        return rel_abs_number >= quant_rel_abs
    elif quant_op == Operator.GREATER:
        return rel_abs_number > quant_rel_abs
    elif quant_op == Operator.LESSEQUAL:
        return rel_abs_number <= quant_rel_abs
    elif quant_op == Operator.LESS:
        return rel_abs_number < quant_rel_abs
    elif quant_op == Operator.EQUAL:
        return rel_abs_number == quant_rel_abs
    else:
        raise ValueError("No valid operator")


def check_interval_condition(x: Union[int, float], left_margin: float,
                             right_margin: float, closure_type: Closure) -> bool:
    """
    Check if the value x meets the condition of the interval [left_margin, right_margin] with closureType

    params:
        :param x: (Union[int, float]) value to check
        :param left_margin: (float) left margin of the interval
        :param right_margin: (float) right margin of the interval
        :param closure_type: (Closure) closure of the interval

    Returns:
        :return: True if the value x meets the condition of the interval
    """
    if closure_type == Closure.openOpen:
        return True if np.issubdtype(type(x), np.number) and ((x > left_margin) & (x < right_margin)) else False
    elif closure_type == Closure.openClosed:
        return True if np.issubdtype(type(x), np.number) and ((x > left_margin) & (x <= right_margin)) else False
    elif closure_type == Closure.closedOpen:
        return True if np.issubdtype(type(x), np.number) and ((x >= left_margin) & (x < right_margin)) else False
    elif closure_type == Closure.closedClosed:
        return True if np.issubdtype(type(x), np.number) and ((x >= left_margin) & (x <= right_margin)) else False


def count_abs_frequency(value, data_dictionary: pd.DataFrame, field: str = None) -> int:
    """
    Count the absolute frequency of a value in all the columns of a dataframe
    If field is not None, the count is done only in the column field

    :param value: value to count
    :param data_dictionary: (pd.DataFrame) dataframe with the data
    :param field: (str) field to count the value

    :return: count: (int) absolute frequency of the value
    """
    if field is not None:
        return data_dictionary[field].value_counts(dropna=False).get(value, 0)
    else:
        count = 0
        for column in data_dictionary:
            count += data_dictionary[column].value_counts(dropna=False).get(value, 0)
        return count


def cast_type_FixValue(data_type_input: DataType = None, fix_value_input = None, data_type_output: DataType = None,
                       fix_value_output = None) -> object:
    """
    Cast the value fix_value_input to the type data_type_output
    and the value fix_value_output to the type data_type_output

    :param data_type_input: data type of the input value
    :param fix_value_input: input value to cast
    :param data_type_output: data type of the output value
    :param fix_value_output: output value to cast

    :return: fix_value_input and fix_value_output casted to the types data_type_input and data_type_output respectively
    """
    if data_type_input is not None and fix_value_input is not None:
        if data_type_input == DataType.STRING:
            fix_value_input = str(fix_value_input)
        elif data_type_input == DataType.TIME:
            fix_value_input = pd.to_datetime(fix_value_input)
        elif data_type_input == DataType.INTEGER:
            fix_value_input = int(fix_value_input)
        elif data_type_input == DataType.DATETIME:
            fix_value_input = pd.to_datetime(fix_value_input)
        elif data_type_input == DataType.BOOLEAN:
            fix_value_input = bool(fix_value_input)
        elif data_type_input == DataType.DOUBLE or data_type_input == DataType.FLOAT:
            fix_value_input = float(fix_value_input)

    if data_type_output is not None and fix_value_output is not None:
        if data_type_output == DataType.STRING:
            fix_value_output = str(fix_value_output)
        elif data_type_output == DataType.TIME:
            fix_value_output = pd.to_datetime(fix_value_output)
        elif data_type_output == DataType.INTEGER:
            fix_value_output = int(fix_value_output)
        elif data_type_output == DataType.DATETIME:
            fix_value_output = pd.to_datetime(fix_value_output)
        elif data_type_output == DataType.BOOLEAN:
            fix_value_output = bool(fix_value_output)
        elif data_type_output == DataType.DOUBLE or data_type_output == DataType.FLOAT:
            fix_value_output = float(fix_value_output)

    return fix_value_input, fix_value_output


def find_closest_value(numeric_values: list, value: Union[int, float]) -> Union[int, float]:
    """
    Find the closest value to a given value in a list of numeric values
    :param numeric_values: list of numeric values
    :param value: (Union[int, float]) value to find the closest value

    :return: closest_value (Union[int, float]): closest value to the given value

    """
    closest_value = None
    min_distance = float('inf')

    for v in numeric_values:
        if v != value and v is not None and np.issubdtype(type(v), np.number):
            distance = abs(v - value)
            if distance < min_distance:
                closest_value = v
                min_distance = distance

    return closest_value


def outlier_closest(data_dictionary: pd.DataFrame, axis_param: int = None, field: str = None):
    """
    Args:
        data_dictionary: dataframe with the data
        axis_param: axis to calculate the outliers
        field: field to calculate the outliers

    Returns: the lower and upper bounds of the outliers
    """

    data_dictionary_copy = data_dictionary.copy()

    threshold = 1.5
    if field is None:
        if axis_param is None:
            data_dictionary_numeric = data_dictionary.select_dtypes(include=[np.number])
            Q1 = data_dictionary_numeric.quantile(0.25)
            Q3 = data_dictionary_numeric.quantile(0.75)
            # Calcula el rango intercuartil
            IQR = Q3 - Q1
            # Define los límites para los outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            return lower_bound, upper_bound

    elif field is not None:
        if axis_param is None or axis_param == 0:
            Q1 = data_dictionary_copy[field].quantile(0.25)
            Q3 = data_dictionary_copy[field].quantile(0.75)
            # Calcula el rango intercuartil
            IQR = Q3 - Q1
            # Define los límites para los outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            return lower_bound, upper_bound
