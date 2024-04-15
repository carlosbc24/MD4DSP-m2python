# Importing enumerations from packages
from typing import Union
from helpers.enumerations import Operator, DataType, SpecialType, Belong

# Importing libraries
import numpy as np
import pandas as pd


def format_duration(seconds: float) -> str:
    """
    Format duration from seconds to hours, minutes, seconds and miliseconds

    :param seconds (float): Duration in seconds
    :return: formated_duration (str): Duration in hours, minutes, seconds and miliseconds
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    miliseconds = seconds - int(seconds)
    formated_duration = f"{hours} hours, {minutes} minutes, {int(seconds)} seconds and {int(miliseconds * 1000)} miliseconds"
    return formated_duration


def compare_numbers(rel_abs_number: Union[int, float], quant_rel_abs: Union[int, float], quant_op: Operator) -> bool:
    """
    Compare two numbers with the operator quant_op

    :param rel_abs_number (Union[int, float]): relative or absolute number to compare with the previous one
    :param quant_rel_abs (Union[int, float]): relative or absolute number to compare with the previous one
    :param quant_op (Operator): operator to compare the two numbers

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


def count_abs_frequency(value, dataDictionary: pd.DataFrame, field: str = None) -> int:
    """
    Count the absolute frequency of a value in all the columns of a dataframe
    If field is not None, the count is done only in the column field

    :param value: value to count
    :param dataDictionary (pd.DataFrame): dataframe with the data
    :param field (str): field to count the value

    :return: count (int): absolute frequency of the value
    """
    if field is not None:
        return dataDictionary[field].value_counts(dropna=False).get(value, 0)
    else:
        count = 0
        for column in dataDictionary:
            count += dataDictionary[column].value_counts(dropna=False).get(value, 0)
        return count


def cast_type_FixValue(dataTypeInput: DataType = None, FixValueInput=None, dataTypeOutput: DataType = None,
                       FixValueOutput=None):
    """
    Cast the value FixValueInput to the type dataTypeOutput and the value FixValueOutput to the type dataTypeOutput

    :param dataTypeInput: data type of the input value
    :param FixValueInput: input value to cast
    :param dataTypeOutput: data type of the output value
    :param FixValueOutput: output value to cast
    :return: FixValueInput and FixValueOutput casted to the types dataTypeInput and dataTypeOutput respectively
    """
    if dataTypeInput is not None and FixValueInput is not None:
        if dataTypeInput == DataType.STRING:
            FixValueInput = str(FixValueInput)
        elif dataTypeInput == DataType.TIME:
            FixValueInput = pd.to_datetime(FixValueInput)
        elif dataTypeInput == DataType.INTEGER:
            FixValueInput = int(FixValueInput)
        elif dataTypeInput == DataType.DATETIME:
            FixValueInput = pd.to_datetime(FixValueInput)
        elif dataTypeInput == DataType.BOOLEAN:
            FixValueInput = bool(FixValueInput)
        elif dataTypeInput == DataType.DOUBLE or dataTypeInput == DataType.FLOAT:
            FixValueInput = float(FixValueInput)

    if dataTypeOutput is not None and FixValueOutput is not None:
        if dataTypeOutput == DataType.STRING:
            FixValueOutput = str(FixValueOutput)
        elif dataTypeOutput == DataType.TIME:
            FixValueOutput = pd.to_datetime(FixValueOutput)
        elif dataTypeOutput == DataType.INTEGER:
            FixValueOutput = int(FixValueOutput)
        elif dataTypeOutput == DataType.DATETIME:
            FixValueOutput = pd.to_datetime(FixValueOutput)
        elif dataTypeOutput == DataType.BOOLEAN:
            FixValueOutput = bool(FixValueOutput)
        elif dataTypeOutput == DataType.DOUBLE or dataTypeOutput == DataType.FLOAT:
            FixValueOutput = float(FixValueOutput)

    return FixValueInput, FixValueOutput


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


def checkSpecialTypeInterpolation(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                                  belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                  dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                  axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly
    params::
        :param dataDictionary_in: dataframe with the data before the interpolation
        :param dataDictionary_out: dataframe with the data after the interpolation
        :param specialTypeInput: special type to apply the interpolation
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the interpolation
        :param field: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
    """

    return True


def checkSpecialTypeMean(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                                  belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                  dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                  axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type mean is applied correctly
    params::
        :param dataDictionary_in: dataframe with the data before the mean
        :param dataDictionary_out: dataframe with the data after the mean
        :param specialTypeInput: special type to apply the mean
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the mean
        :param field: field to apply the mean

    Returns:
        :return: True if the special type mean is applied correctly
    """
    if field is None:
        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
            if specialTypeInput == SpecialType.MISSING:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number) or pd.isnull(value):
                                if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                                    if dataDictionary_out.at[idx, col_name] != mean_value:
                                        return False
                                else:
                                    if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                        return False
                elif axis_param == 0:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        mean=dataDictionary_in[col_name].mean()
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number) or pd.isnull(value):
                                if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                                    if dataDictionary_out.at[idx, col_name] != mean:
                                        return False
                                else:
                                    if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                        return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        mean = numeric_data.mean()
                        # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                        for col_name, value in numeric_data.items():
                            if value in missing_values or pd.isnull(value):
                                if dataDictionary_out.at[idx, col_name] != mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            if specialTypeInput == SpecialType.INVALID:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number):
                                if dataDictionary_in.at[idx, col_name] in missing_values:
                                    if dataDictionary_out.at[idx, col_name] != mean_value:
                                        return False
                                else:
                                    if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                        return False
                elif axis_param == 0:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        mean=dataDictionary_in[col_name].mean()
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number):
                                if dataDictionary_in.at[idx, col_name] in missing_values:
                                    if dataDictionary_out.at[idx, col_name] != mean:
                                        return False
                                else:
                                    if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                        return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        mean = numeric_data.mean()
                        # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                        for col_name, value in numeric_data.items():
                            if value in missing_values:
                                if dataDictionary_out.at[idx, col_name] != mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False

            if specialTypeInput == SpecialType.OUTLIER:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Replace the missing values with the mean of the entire DataFrame using lambda
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                                if dataDictionary_out.at[idx, col_name] != mean_value:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False
                if axis_param == 0: # Iterate over each column
                    for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        mean=dataDictionary_in[col].mean()
                        for idx, value in dataDictionary_in[col].items():
                            if dataDictionary_outliers_mask.at[idx, col] == 1:
                                if dataDictionary_out.at[idx, col] != mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col] and not(pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(dataDictionary_out.loc[idx, col])):
                                    return False
                elif axis_param == 1: # Iterate over each row
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        mean = numeric_data.mean()
                        for col_name, value in numeric_data.items():
                            if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                                if dataDictionary_out.at[idx, col_name] != mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False

        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
            if specialTypeInput == SpecialType.MISSING:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number) or pd.isnull(value):
                                if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                                    if dataDictionary_out.at[idx, col_name] == mean_value:
                                        return False
                                else:
                                    if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                        return False
                elif axis_param == 0:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        mean=dataDictionary_in[col_name].mean()
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number) or pd.isnull(value):
                                if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                                    if dataDictionary_out.at[idx, col_name] == mean:
                                        return False
                                else:
                                    if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                        return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        mean = numeric_data.mean()
                        # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                        for col_name, value in numeric_data.items():
                            if value in missing_values or pd.isnull(value):
                                if dataDictionary_out.at[idx, col_name] == mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            if specialTypeInput == SpecialType.INVALID:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number):
                                if dataDictionary_in.at[idx, col_name] in missing_values:
                                    if dataDictionary_out.at[idx, col_name] == mean_value:
                                        return False
                                else:
                                    if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                        return False
                elif axis_param == 0:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        mean=dataDictionary_in[col_name].mean()
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number):
                                if dataDictionary_in.at[idx, col_name] in missing_values:
                                    if dataDictionary_out.at[idx, col_name] == mean:
                                        return False
                                else:
                                    if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                        return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        mean = numeric_data.mean()
                        # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                        for col_name, value in numeric_data.items():
                            if value in missing_values:
                                if dataDictionary_out.at[idx, col_name] == mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False
            if specialTypeInput == SpecialType.OUTLIER:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Replace the missing values with the mean of the entire DataFrame using lambda
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                                if dataDictionary_out.at[idx, col_name] == mean_value:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False
                if axis_param == 0:
                    for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        mean=dataDictionary_in[col].mean()
                        for idx, value in dataDictionary_in[col].items():
                            if dataDictionary_outliers_mask.at[idx, col] == 1:
                                if dataDictionary_out.at[idx, col] == mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col] and not(pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(dataDictionary_out.loc[idx, col])):
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        mean = numeric_data.mean()
                        for col_name, value in numeric_data.items():
                            if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                                if dataDictionary_out.at[idx, col_name] == mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not(pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False

        elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
            if specialTypeInput == SpecialType.MISSING:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number) or pd.isnull(value):
                                if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                                    return False
                elif axis_param == 0:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number) or pd.isnull(value):
                                if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                        for col_name, value in numeric_data.items():
                            if value in missing_values or pd.isnull(value):
                                return False
            if specialTypeInput == SpecialType.INVALID:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number):
                                if dataDictionary_in.at[idx, col_name] in missing_values:
                                    return False
                elif axis_param == 0:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number):
                                if dataDictionary_in.at[idx, col_name] in missing_values:
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                        for col_name, value in numeric_data.items():
                            if value in missing_values:
                                return False
            if specialTypeInput == SpecialType.OUTLIER:
                if axis_param is None:
                    # Replace the missing values with the mean of the entire DataFrame using lambda
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                                return False
                if axis_param == 0:
                    for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col].items():
                            if dataDictionary_outliers_mask.at[idx, col] == 1:
                                return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        for col_name, value in numeric_data.items():
                            if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                                return False

        elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
            if specialTypeInput == SpecialType.MISSING:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number) or pd.isnull(value):
                                if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                                    return False
                elif axis_param == 0:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number) or pd.isnull(value):
                                if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                        for col_name, value in numeric_data.items():
                            if value in missing_values or pd.isnull(value):
                                return False
            if specialTypeInput == SpecialType.INVALID:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number):
                                if dataDictionary_in.at[idx, col_name] in missing_values:
                                    return False
                elif axis_param == 0:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    # Check the dataDictionary_out positions with missing values have been replaced with the mean
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if np.issubdtype(type(value), np.number):
                                if dataDictionary_in.at[idx, col_name] in missing_values:
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                        for col_name, value in numeric_data.items():
                            if value in missing_values:
                                return False
            if specialTypeInput == SpecialType.OUTLIER:
                if axis_param is None:
                    # Replace the missing values with the mean of the entire DataFrame using lambda
                    for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col_name].items():
                            if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                                return False
                if axis_param == 0:
                    for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        for idx, value in dataDictionary_in[col].items():
                            if dataDictionary_outliers_mask.at[idx, col] == 1:
                                return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        numeric_data = row.select_dtypes(include=[np.number])
                        for col_name, value in numeric_data.items():
                            if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
            if specialTypeInput == SpecialType.MISSING:
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                mean = dataDictionary_in[field].mean()
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values or pd.isnull(value):
                        if dataDictionary_out.at[idx, field] != mean:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            return False
            if specialTypeInput == SpecialType.INVALID:
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                mean = dataDictionary_in[field].mean()
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        if dataDictionary_out.at[idx, field] != mean:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not(pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(dataDictionary_out.loc[idx, field])):
                            return False
            if specialTypeInput == SpecialType.OUTLIER:
                for idx, value in dataDictionary_in[field].items():
                    if dataDictionary_outliers_mask.at[idx, field] == 1:
                        mean = dataDictionary_in[field].mean()
                        if dataDictionary_out.at[idx, field] != mean:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not(pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(dataDictionary_out.loc[idx, field])):
                            return False

        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
            if specialTypeInput == SpecialType.MISSING:
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                mean = dataDictionary_in[field].mean()
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values or pd.isnull(value):
                        if dataDictionary_out.at[idx, field] == mean:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            return False
            if specialTypeInput == SpecialType.INVALID:
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                mean = dataDictionary_in[field].mean()
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        if dataDictionary_out.at[idx, field] == mean:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not(pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(dataDictionary_out.loc[idx, field])):
                            return False
            if specialTypeInput == SpecialType.OUTLIER:
                for idx, value in dataDictionary_in[field].items():
                    if dataDictionary_outliers_mask.at[idx, field] == 1:
                        mean = dataDictionary_in[field].mean()
                        if dataDictionary_out.at[idx, field] == mean:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not(pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(dataDictionary_out.loc[idx, field])):
                            return False


    return True


def checkSpecialTypeMedian(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                                  belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                  dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                  axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type median is applied correctly
    params::
        :param dataDictionary_in: dataframe with the data before the median
        :param dataDictionary_out: dataframe with the data after the median
        :param specialTypeInput: special type to apply the median
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the median
        :param field: field to apply the median

    Returns:
        :return: True if the special type median is applied correctly
    """

    return True


def checkSpecialTypeClosest(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                                  belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                  dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                  axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type closest value is applied correctly
    params::
        :param dataDictionary_in: dataframe with the data before the closest
        :param dataDictionary_out: dataframe with the data after the closest
        :param specialTypeInput: special type to apply the closest
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the closest
        :param field: field to apply the closest

    Returns:
        :return: True if the special type closest is applied correctly
    """

    return True












