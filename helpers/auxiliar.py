# Importing enumerations from packages
from typing import Union
from helpers.enumerations import Operator, DataType, Closure

# Importing libraries
import numpy as np
import pandas as pd


def format_duration(seconds: float) -> str:
    """
    Format duration from seconds to hours, minutes, seconds and milliseconds

    :param seconds (float): Duration in seconds
    :return: formated_duration (str): Duration in hours, minutes, seconds and milliseconds
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    miliseconds = seconds - int(seconds)
    formated_duration = f"{hours} hours, {minutes} minutes, {int(seconds)} seconds and {int(miliseconds * 1000)} milliseconds"
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


def check_interval_condition(x: Union[int, float], leftMargin: float, rightMargin: float, closureType: Closure) -> bool:
    """
    Check if the value x meets the condition of the interval [leftMargin, rightMargin] with closureType

    params:
        :param x: (Union[int, float]) value to check
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval

    Returns:
        :return: True if the value x meets the condition of the interval
    """
    if closureType == Closure.openOpen:
        return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x < rightMargin)) else False
    elif closureType == Closure.openClosed:
        return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x <= rightMargin)) else False
    elif closureType == Closure.closedOpen:
        return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x < rightMargin)) else False
    elif closureType == Closure.closedClosed:
        return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x <= rightMargin)) else False


def count_abs_frequency(value, dataDictionary: pd.DataFrame, field: str = None) -> int:
    """
    Count the absolute frequency of a value in all the columns of a dataframe
    If field is not None, the count is done only in the column field

    :param value: value to count
    :param dataDictionary: (pd.DataFrame) dataframe with the data
    :param field: (str) field to count the value

    :return: count: (int) absolute frequency of the value
    """
    if field is not None:
        return dataDictionary[field].value_counts(dropna=False).get(value, 0)
    else:
        count = 0
        for column in dataDictionary:
            count += dataDictionary[column].value_counts(dropna=False).get(value, 0)
        return count


def cast_type_FixValue(dataTypeInput: DataType = None, fixValueInput=None, dataTypeOutput: DataType = None,
                       fixValueOutput=None):
    """
    Cast the value FixValueInput to the type dataTypeOutput and the value FixValueOutput to the type dataTypeOutput

    :param dataTypeInput: data type of the input value
    :param fixValueInput: input value to cast
    :param dataTypeOutput: data type of the output value
    :param fixValueOutput: output value to cast

    :return: FixValueInput and FixValueOutput casted to the types dataTypeInput and dataTypeOutput respectively
    """
    if dataTypeInput is not None and fixValueInput is not None:
        if dataTypeInput == DataType.STRING:
            fixValueInput = str(fixValueInput)
        elif dataTypeInput == DataType.TIME:
            fixValueInput = pd.to_datetime(fixValueInput)
        elif dataTypeInput == DataType.INTEGER:
            fixValueInput = int(fixValueInput)
        elif dataTypeInput == DataType.DATETIME:
            fixValueInput = pd.to_datetime(fixValueInput)
        elif dataTypeInput == DataType.BOOLEAN:
            fixValueInput = bool(fixValueInput)
        elif dataTypeInput == DataType.DOUBLE or dataTypeInput == DataType.FLOAT:
            fixValueInput = float(fixValueInput)

    if dataTypeOutput is not None and fixValueOutput is not None:
        if dataTypeOutput == DataType.STRING:
            fixValueOutput = str(fixValueOutput)
        elif dataTypeOutput == DataType.TIME:
            fixValueOutput = pd.to_datetime(fixValueOutput)
        elif dataTypeOutput == DataType.INTEGER:
            fixValueOutput = int(fixValueOutput)
        elif dataTypeOutput == DataType.DATETIME:
            fixValueOutput = pd.to_datetime(fixValueOutput)
        elif dataTypeOutput == DataType.BOOLEAN:
            fixValueOutput = bool(fixValueOutput)
        elif dataTypeOutput == DataType.DOUBLE or dataTypeOutput == DataType.FLOAT:
            fixValueOutput = float(fixValueOutput)

    return fixValueInput, fixValueOutput


def find_closest_value(numeric_values: list, value: Union[int, float]) -> Union[int, float]:
    """
    Find the closest value to a given value in a list of numeric values
    :param numeric_values: list of numeric values
    :param value (Union[int, float]): value to find the closest value

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


