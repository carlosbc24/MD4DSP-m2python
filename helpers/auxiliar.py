# Importing enumerations from packages
from typing import Union
from helpers.enumerations import Operator, DataType, SpecialType, DerivedType, Belong, Operator, DataType

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
    params:
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
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result=checkInterpolationBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result=checkInterpolationBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result=checkInterpolationNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result=checkInterpolationNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)

    return True if result else False


def checkInterpolationBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are BELONG
    params:
        :param dataDictionary_in: dataframe with the data before the interpolation
        :param dataDictionary_out: dataframe with the data after the interpolation
        :param specialTypeInput: special type to apply the interpolation
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the interpolation
        :param field: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
        """
    dataDictionary_in_copy = dataDictionary_in.copy()
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    result = (dataDictionary_in[col_name].apply(lambda x: np.nan if x in missing_values else x).
                                interpolate(method='linear', limit_direction='both').equals(dataDictionary_out[col_name]))
                    if result is False:
                        return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    result = (numeric_data[row].apply(lambda x: np.nan if x in missing_values else x).
                                interpolate(method='linear', limit_direction='both').equals(dataDictionary_out[row]))
                    if result is False:
                        return False

        if specialTypeInput == SpecialType.INVALID:
            # Applies the linear interpolation in the DataFrame
            if axis_param == 0:
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        dataDictionary_in_copy[col] = (dataDictionary_in[col].apply(lambda x: np.nan if x in missing_values else x).
                                                       interpolate(method='linear', limit_direction='both'))
                # Iterate over each column
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    # For each index in the column
                    for idx in dataDictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(dataDictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            dataDictionary_in_copy.at[idx, col] = dataDictionary_in.at[idx, col]
                result=dataDictionary_in_copy.equals(dataDictionary_out)
                if result is False:
                    return False


    return True


def checkInterpolationBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                   specialTypeInput: SpecialType,
                                   dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                   axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframes when belongOp_in is BELONG and belongOp_out is NOTBELONG
    params:
        :param dataDictionary_in: dataframe with the data before the interpolation
        :param dataDictionary_out: dataframe with the data after the interpolation
        :param specialTypeInput: special type to apply the interpolation
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the interpolation
        :param field: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
        """


    return True


def checkInterpolationNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                   specialTypeInput: SpecialType,
                                   dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                   axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframes when belongOp_in is NOTBELONG and belongOp_out is BELONG
    params:
        :param dataDictionary_in: dataframe with the data before the interpolation
        :param dataDictionary_out: dataframe with the data after the interpolation
        :param specialTypeInput: special type to apply the interpolation
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the interpolation
        :param field: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
        """


    return True


def checkInterpolationNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                   specialTypeInput: SpecialType,
                                   dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                   axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are NOTBELONG
    params:
        :param dataDictionary_in: dataframe with the data before the interpolation
        :param dataDictionary_out: dataframe with the data after the interpolation
        :param specialTypeInput: special type to apply the interpolation
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
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result=checkMeanBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result=checkMeanBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result=checkMeanNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result=checkMeanNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)

    return True if result else False


def checkMeanBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type mean is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are BELONG
    params::
        :param dataDictionary_in: dataframe with the data before the mean
        :param dataDictionary_out: dataframe with the data after the mean
        :param specialTypeInput: special type to apply the mean
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the mean
        :param field: field to apply the mean

    Returns:
        :return: True if the special type mean is applied correctly
    """
    if field is None:
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
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                if dataDictionary_out.at[idx, col_name] != mean_value:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = dataDictionary_in[col_name].mean()
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
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
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                        dataDictionary_out.loc[idx, col_name])):
                                    return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = dataDictionary_in[col_name].mean()
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                if dataDictionary_out.at[idx, col_name] != mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                        dataDictionary_out.loc[idx, col_name])):
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
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
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
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                return False
            if axis_param == 0:  # Iterate over each column
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = dataDictionary_in[col].mean()
                    for idx, value in dataDictionary_in[col].items():
                        if dataDictionary_outliers_mask.at[idx, col] == 1:
                            if dataDictionary_out.at[idx, col] != mean:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                return False
            elif axis_param == 1:  # Iterate over each row
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    mean = numeric_data.mean()
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            if dataDictionary_out.at[idx, col_name] != mean:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and
                                     pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

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
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                        return False
        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    mean = dataDictionary_in[field].mean()
                    if dataDictionary_out.at[idx, field] != mean:
                        return False
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                        return False

    return True


def checkMeanBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type mean is applied correctly when the input and output dataframes when belongOp_in is Belong and belongOp_out is NOTBELONG
    params::
        :param dataDictionary_in: dataframe with the data before the mean
        :param dataDictionary_out: dataframe with the data after the mean
        :param specialTypeInput: special type to apply the mean
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the mean
        :param field: field to apply the mean

    Returns:
        :return: True if the special type mean is applied correctly
    """
    if field is None:
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
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                if dataDictionary_out.at[idx, col_name] == mean_value:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = dataDictionary_in[col_name].mean()
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
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
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (pd.isnull(dataDictionary_in.loc[idx, col_name])
                                                            and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = dataDictionary_in[col_name].mean()
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                if dataDictionary_out.at[idx, col_name] == mean:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (pd.isnull(dataDictionary_in.loc[idx, col_name])
                                                            and pd.isnull(dataDictionary_out.loc[idx, col_name])):
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
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
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
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                return False
            if axis_param == 0:
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = dataDictionary_in[col].mean()
                    for idx, value in dataDictionary_in[col].items():
                        if dataDictionary_outliers_mask.at[idx, col] == 1:
                            if dataDictionary_out.at[idx, col] == mean:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
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
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

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
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                        return False
        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    mean = dataDictionary_in[field].mean()
                    if dataDictionary_out.at[idx, field] == mean:
                        return False
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                        return False

    return True


def checkMeanNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type mean is applied correctly when the input and output dataframes when belongOp_in is NotBelong and belongOp_out is BELONG
    params::
        :param dataDictionary_in: dataframe with the data before the mean
        :param dataDictionary_out: dataframe with the data after the mean
        :param specialTypeInput: special type to apply the mean
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the mean
        :param field: field to apply the mean

    Returns:
        :return: True if the special type mean is applied correctly
    """
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
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

        if specialTypeInput == SpecialType.MISSING:
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values or pd.isnull(value):
                    return False
        if specialTypeInput == SpecialType.INVALID:
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values:
                    return False
        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    return False

    return True


def checkMeanNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type mean is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are NOTBELONG
    params::
        :param dataDictionary_in: dataframe with the data before the mean
        :param dataDictionary_out: dataframe with the data after the mean
        :param specialTypeInput: special type to apply the mean
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the mean
        :param field: field to apply the mean

    Returns:
        :return: True if the special type mean is applied correctly
    """
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the mean
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
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

        if specialTypeInput == SpecialType.MISSING:
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values or pd.isnull(value):
                    return False
        if specialTypeInput == SpecialType.INVALID:
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values:
                    return False
        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
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
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result=checkMedianBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result=checkMedianBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result=checkMedianNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result=checkMedianNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                        specialTypeInput=specialTypeInput,
                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                        missing_values=missing_values, axis_param=axis_param,
                                        field=field)

    return True if result else False


def checkMedianBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type median is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are BELONG
    params::
        :param dataDictionary_in: dataframe with the data before the median
        :param dataDictionary_out: dataframe with the data after the median
        :param specialTypeInput: special type to apply the median
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the median
        :param field: field to apply the median

    Returns:
        :return: True if the special type median is applied correctly
    """
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                if dataDictionary_out.at[idx, col_name] != median_value:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    median = dataDictionary_in[col_name].median()
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                if dataDictionary_out.at[idx, col_name] != median:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    median = numeric_data.median()
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.at[idx, col_name] != median:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                if dataDictionary_out.at[idx, col_name] != median_value:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                        dataDictionary_out.loc[idx, col_name])):
                                    return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    median = dataDictionary_in[col_name].median()
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                if dataDictionary_out.at[idx, col_name] != median:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                        dataDictionary_out.loc[idx, col_name])):
                                    return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    median = numeric_data.median()
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values:
                            if dataDictionary_out.at[idx, col_name] != median:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                return False

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Replace the missing values with the median of the entire DataFrame using lambda
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            if dataDictionary_out.at[idx, col_name] != median_value:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                return False
            if axis_param == 0:  # Iterate over each column
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    median = dataDictionary_in[col].median()
                    for idx, value in dataDictionary_in[col].items():
                        if dataDictionary_outliers_mask.at[idx, col] == 1:
                            if dataDictionary_out.at[idx, col] != median:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                return False
            elif axis_param == 1:  # Iterate over each row
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    median = numeric_data.median()
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            if dataDictionary_out.at[idx, col_name] != median:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and
                                     pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if specialTypeInput == SpecialType.MISSING:
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            median = dataDictionary_in[field].median()
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values or pd.isnull(value):
                    if dataDictionary_out.at[idx, field] != median:
                        return False
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False
        if specialTypeInput == SpecialType.INVALID:
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            median = dataDictionary_in[field].median()
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values:
                    if dataDictionary_out.at[idx, field] != median:
                        return False
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                        return False
        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    median = dataDictionary_in[field].median()
                    if dataDictionary_out.at[idx, field] != median:
                        return False
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                        return False

    return True


def checkMedianBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type median is applied correctly when the input and output dataframes when belongOp_in is Belong and belongOp_out is NOTBELONG
    params::
        :param dataDictionary_in: dataframe with the data before the median
        :param dataDictionary_out: dataframe with the data after the median
        :param specialTypeInput: special type to apply the median
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the median
        :param field: field to apply the median

    Returns:
        :return: True if the special type median is applied correctly
    """
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                if dataDictionary_out.at[idx, col_name] == median_value:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    median = dataDictionary_in[col_name].median()
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                if dataDictionary_out.at[idx, col_name] == median:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    median = numeric_data.median()
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.at[idx, col_name] == median:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                if dataDictionary_out.at[idx, col_name] == median_value:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (pd.isnull(dataDictionary_in.loc[idx, col_name])
                                                            and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    median = dataDictionary_in[col_name].median()
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                if dataDictionary_out.at[idx, col_name] == median:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (pd.isnull(dataDictionary_in.loc[idx, col_name])
                                                            and pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                    return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    median = numeric_data.median()
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values:
                            if dataDictionary_out.at[idx, col_name] == median:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                return False
        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Replace the missing values with the median of the entire DataFrame using lambda
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            if dataDictionary_out.at[idx, col_name] == median_value:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                return False
            if axis_param == 0:
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    median = dataDictionary_in[col].median()
                    for idx, value in dataDictionary_in[col].items():
                        if dataDictionary_outliers_mask.at[idx, col] == 1:
                            if dataDictionary_out.at[idx, col] == median:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    median = numeric_data.median()
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            if dataDictionary_out.at[idx, col_name] == median:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if specialTypeInput == SpecialType.MISSING:
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            median = dataDictionary_in[field].median()
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values or pd.isnull(value):
                    if dataDictionary_out.at[idx, field] == median:
                        return False
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False
        if specialTypeInput == SpecialType.INVALID:
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            median = dataDictionary_in[field].median()
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values:
                    if dataDictionary_out.at[idx, field] == median:
                        return False
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                        return False
        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    median = dataDictionary_in[field].median()
                    if dataDictionary_out.at[idx, field] == median:
                        return False
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                        return False

    return True


def checkMedianNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type median is applied correctly when the input and output dataframes when belongOp_in is NotBelong and belongOp_out is BELONG
    params::
        :param dataDictionary_in: dataframe with the data before the median
        :param dataDictionary_out: dataframe with the data after the median
        :param specialTypeInput: special type to apply the median
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the median
        :param field: field to apply the median

    Returns:
        :return: True if the special type median is applied correctly
    """
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values:
                            return False
        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                # Replace the missing values with the median of the entire DataFrame using lambda
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

        if specialTypeInput == SpecialType.MISSING:
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values or pd.isnull(value):
                    return False
        if specialTypeInput == SpecialType.INVALID:
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values:
                    return False
        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    return False

    return True


def checkMedianNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                            dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type median is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are NOTBELONG
    params::
        :param dataDictionary_in: dataframe with the data before the median
        :param dataDictionary_out: dataframe with the data after the median
        :param specialTypeInput: special type to apply the median
        :param dataDictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the median
        :param field: field to apply the median

    Returns:
        :return: True if the special type median is applied correctly
    """
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    dataDictionary_in.at[idx, col_name]):
                                return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                return False
            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the dataDictionary_out positions with missing values have been replaced with the median
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if dataDictionary_in.at[idx, col_name] in missing_values:
                                return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row.select_dtypes(include=[np.number])
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
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

        if specialTypeInput == SpecialType.MISSING:
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values or pd.isnull(value):
                    return False
        if specialTypeInput == SpecialType.INVALID:
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values:
                    return False
        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    return False

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


def checkMostFrequentBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                  specialTypeInput: SpecialType, missing_values: list, axis_param: int, field: str) -> bool:
    """
    Check if the special type most frequent is applied correctly when the input and output dataframes when belongOp_in is Belong and belongOp_out is BELONG

    params:

    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the most frequent
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the most frequent
    :param specialTypeInput: (SpecialType) Special type to apply the most frequent
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the most frequent
    :param field: (str) Field to apply the most frequent

    :return: True if the special type most frequent is applied correctly, False otherwise
    """
    if field is None:
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_name] != most_frequent:
                                return False
                        else:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False
            if specialTypeInput == SpecialType.INVALID:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_name] != most_frequent:
                                return False
                        else:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                    dataDictionary_out.loc[row_index, column_name])):
                                return False
        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                # Instead of iterating over the columns, we iterate over the rows to check the derived type
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
            if specialTypeInput == SpecialType.INVALID:
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                    dataDictionary_out.loc[row_index, column_index])):
                                return False
        elif axis_param is None:
            most_frequent = dataDictionary_in.stack().value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
            if specialTypeInput == SpecialType.INVALID:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                    dataDictionary_out.loc[row_index, column_index])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            most_frequent = dataDictionary_in[field].value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values or pd.isnull(value):
                        if dataDictionary_out.loc[idx, field] != most_frequent:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            return False
            elif specialTypeInput == SpecialType.INVALID:
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        if dataDictionary_out.loc[idx, field] != most_frequent:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            return False

    return True


def checkMostFrequentBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     specialTypeInput: SpecialType, missing_values: list, axis_param: int, field) -> bool:
    """
    Check if the special type most frequent is applied correctly when the input and output dataframes when belongOp_in is Belong and belongOp_out is NOTBELONG

    params:
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the most frequent
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the most frequent
    :param specialTypeInput: (SpecialType) Special type to apply the most frequent
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the most frequent
    :param field: (str) Field to apply the most frequent

    :return: True if the special type most frequent is applied correctly, False otherwise
    """
    if field is None:
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_name] == most_frequent:
                                return False
                        else:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False
            if specialTypeInput == SpecialType.INVALID:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_name] == most_frequent:
                                return False
                        else:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_name])):
                                return False
        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                # Instead of iterating over the columns, we iterate over the rows to check the derived type
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_index] == most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
            if specialTypeInput == SpecialType.INVALID:
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_index] == most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_index])):
                                return False
        elif axis_param is None:
            most_frequent = dataDictionary_in.stack().value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_index] == most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
            if specialTypeInput == SpecialType.INVALID:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_index] == most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_index])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            most_frequent = dataDictionary_in[field].value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values or pd.isnull(value):
                        if dataDictionary_out.loc[idx, field] == most_frequent:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            return False
            elif specialTypeInput == SpecialType.INVALID:
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        if dataDictionary_out.loc[idx, field] == most_frequent:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            return False

    return True



def checkMostFrequentNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     specialTypeInput: SpecialType, missing_values: list, axis_param: int, field) -> bool:
    """
    Check if the special type most frequent is applied correctly when the input and output dataframes when belongOp_in is NOTBELONG and belongOp_out is BELONG

    params:
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the most frequent
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the most frequent
    :param specialTypeInput: (SpecialType) Special type to apply the most frequent
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the most frequent
    :param field: (str) Field to apply the most frequent

    :return: True if the special type most frequent is applied correctly, False otherwise
    """
    if field is None:
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False
            if specialTypeInput == SpecialType.INVALID:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            return False
        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                # Instead of iterating over the columns, we iterate over the rows to check the derived type
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False
            if specialTypeInput == SpecialType.INVALID:
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values:
                            return False
        elif axis_param is None:
            most_frequent = dataDictionary_in.stack().value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False
            if specialTypeInput == SpecialType.INVALID:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            most_frequent = dataDictionary_in[field].value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values or pd.isnull(value):
                        return False
            elif specialTypeInput == SpecialType.INVALID:
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        return False

    return True


def checkMostFrequentNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        specialTypeInput: SpecialType, missing_values: list, axis_param: int, field) -> bool:
    """
    Check if the special type most frequent is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are NOTBELONG

    params:
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the most frequent
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the most frequent
    :param specialTypeInput: (SpecialType) Special type to apply the most frequent
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the most frequent
    :param field: (str) Field to apply the most frequent

    :return: True if the special type most frequent is applied correctly, False otherwise
    """
    if field is None:
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False
            if specialTypeInput == SpecialType.INVALID:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            return False
        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                # Instead of iterating over the columns, we iterate over the rows to check the derived type
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False
            if specialTypeInput == SpecialType.INVALID:
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values:
                            return False
        elif axis_param is None:
            most_frequent = dataDictionary_in.stack().value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False
            if specialTypeInput == SpecialType.INVALID:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            most_frequent = dataDictionary_in[field].value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values or pd.isnull(value):
                        return False
            elif specialTypeInput == SpecialType.INVALID:
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        return False

    return True


def checkDerivedTypeMostFrequent(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                                  belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                  missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
    """
    Check if the derived type most frequent value is applied correctly
    params:
        :param dataDictionary_in: dataframe with the data before the MostFrequent
        :param dataDictionary_out: dataframe with the data after the MostFrequent
        :param specialTypeInput: special type to apply the MostFrequent
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to apply the MostFrequent
        :param field: field to apply the MostFrequent

    Returns:
        :return: True if the derived type most frequent value is applied correctly
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result=checkMostFrequentBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                             specialTypeInput=specialTypeInput, missing_values=missing_values,
                                             axis_param=axis_param, field=field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result=checkMostFrequentBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result=checkMostFrequentNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result=checkMostFrequentNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                   specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                   axis_param=axis_param, field=field)

    return True if result else False


def checkDerivedTypePreviousBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                         specialTypeInput: SpecialType, missing_values: list, axis_param: int, field: str) -> bool:
    """
    Check if the derived type previous value is applied correctly when the input and output dataframes when belongOp_in is Belong and belongOp_out is BELONG

    params:
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the previous
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the previous
    :param specialTypeInput: (SpecialType) Special type to apply the previous
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the previous
    :param field: (str) Field to apply the previous

    :return: True if the derived type previous value is applied correctly, False otherwise
    """
    if field is None:
        # Check the previous value of the missing values
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                # Manual check of the previous operacion to the missing values in the columns
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == 0:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index - 1, column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False

            else:  # SPECIAL_TYPE is INVALID
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == 0:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index - 1, column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_name])):
                                return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if column_index == 0:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index - 1]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False

            else:  # SPECIAL_TYPE is INVALID
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if column_index == 0:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index - 1]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_index])):
                                return False

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            if specialTypeInput == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            if idx == 0:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False
                else: # missing_values is None
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == 0:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == 0:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not(pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(dataDictionary_out.loc[idx, field])):
                                return False

    return True


def checkDerivedTypePreviousBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                            specialTypeInput: SpecialType, missing_values: list, axis_param: int, field: str) -> bool:
    """
    Check if the derived type previous is applied correctly when the input and output dataframes when belongOp_in is Belong and belongOp_out is NOTBELONG

    params:
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the previous
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the previous
    :param specialTypeInput: (SpecialType) Special type to apply the previous
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the previous
    :param field: (str) Field to apply the previous

    :return: True if the derived type previous is applied correctly, False otherwise
    """
    if field is None:
        # Check the previous value of the missing values
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                # Manual check of the previous operacion to the missing values in the columns
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == 0:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] == dataDictionary_in.loc[
                                    row_index - 1, column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False

            else:  # SPECIAL_TYPE is INVALID
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == 0:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] == dataDictionary_in.loc[
                                    row_index - 1, column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_name])):
                                return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if column_index == 0:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] == dataDictionary_in.loc[
                                    row_index, column_index - 1]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False

            else:  # SPECIAL_TYPE is INVALID
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if column_index == 0:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] == dataDictionary_in.loc[
                                    row_index, column_index - 1]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_index])):
                                return False

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            if specialTypeInput == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            if idx == 0:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] == dataDictionary_in.loc[idx - 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False
                else:  # missing_values is None
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == 0:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] == dataDictionary_in.loc[idx - 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == 0:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] == dataDictionary_in.loc[idx - 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False

    return True


def checkDerivedTypePreviousNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                            specialTypeInput: SpecialType, missing_values: list, axis_param: int,
                                            field: str) -> bool:
    """
    Check if the derived type previous value is applied correctly when the input and output dataframes when belongOp_in is NOTBELONG and belongOp_out is BELONG

    params:
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the Previous
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the Previous
    :param specialTypeInput: (SpecialType) Special type to apply the Previous
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the Previous
    :param field: (str) Field to apply the Previous

    :return: True if the derived type previous value is applied correctly, False otherwise
    """
    if field is None:
        # Check the previous value of the missing values
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                # Manual check of the previous operacion to the missing values in the columns
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False

            else:  # SPECIAL_TYPE is INVALID
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False

            else:  # SPECIAL_TYPE is INVALID
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            if specialTypeInput == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            return False
                else:  # missing_values is None
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

    return True


def checkDerivedTypePreviousNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                               specialTypeInput: SpecialType, missing_values: list, axis_param: int,
                                                  field: str) -> bool:
    """
    Check if the derived type previous value is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are NOTBELONG

    params:
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the Previous
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the Previous
    :param specialTypeInput: (SpecialType) Special type to apply the Previous
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the Previous
    :param field: (str) Field to apply the Previou

    :return: True if the derived type previous value is applied correctly, False otherwise
    """
    if field is None:
        # Check the previous value of the missing values
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                # Manual check of the previous operacion to the missing values in the columns
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False

            else:  # SPECIAL_TYPE is INVALID
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False

            else:  # SPECIAL_TYPE is INVALID
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            if specialTypeInput == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            return False
                else:  # missing_values is None
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

    return True


def checkDerivedTypePrevious(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                                  belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                  missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
    """
    Check if the derived type previous value is applied correctly
    params:
        :param dataDictionary_in: dataframe with the data before the Previous
        :param dataDictionary_out: dataframe with the data after th Previous
        :param specialTypeInput: special type to apply the Previous
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to apply the Previous
        :param field: field to apply the Previous

    Returns:
        :return: True if the derived type previous value is applied correctly
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result=checkDerivedTypePreviousBelongBelong(dataDictionary_in=dataDictionary_in,
                                                    dataDictionary_out=dataDictionary_out,
                                                    specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                    axis_param=axis_param, field=field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result=checkDerivedTypePreviousBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                       dataDictionary_out=dataDictionary_out,
                                                       specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                       axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result=checkDerivedTypePreviousNotBelongBelong(dataDictionary_in=dataDictionary_in,
                                                       dataDictionary_out=dataDictionary_out,
                                                       specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                       axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result=checkDerivedTypePreviousNotBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                          dataDictionary_out=dataDictionary_out,
                                                          specialTypeInput=specialTypeInput,
                                                          missing_values=missing_values, axis_param=axis_param,
                                                          field=field)

    return True if result else False


def checkDerivedTypeNextBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     specialTypeInput: SpecialType, missing_values: list, axis_param: int, field: str) -> bool:
    """
    Check if the derived type next value is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are BELONG
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the next
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the next
    :param specialTypeInput: (SpecialType) Special type to apply the next
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the next
    :param field: (str) Field to apply the next

    :return: True if the derived type next value is applied correctly, False otherwise
    """
    if field is None:
        # Define the lambda function to replace the values within missing values by the value of the next position
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index + 1, column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False

            else:  # SPECIAL_TYPE is INVALID
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index + 1, column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_name])):
                                return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if column_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index + 1]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
            else:  # SPECIAL_TYPE is INVALID
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if column_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index + 1]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_index])):
                                return False

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            if specialTypeInput == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            if idx == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx + 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

                else:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx + 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx + 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not(pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(dataDictionary_out.loc[idx, field])):
                                return False

    return True


def checkDerivedTypeNextBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     specialTypeInput: SpecialType, missing_values: list, axis_param: int, field: str) -> bool:
    """
    Check if the derived type next value is applied correctly when the input and output dataframes when belongOp_in is BELONG and belongOp_out is NOTBELONG
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the next
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the next
    :param specialTypeInput: (SpecialType) Special type to apply the next
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the next
    :param field: (str) Field to apply the next

    :return: True if the derived type next value is applied correctly, False otherwise
    """
    if field is None:
        # Define the lambda function to replace the values within missing values by the value of the next position
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] == dataDictionary_in.loc[
                                    row_index + 1, column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False

            else:  # SPECIAL_TYPE is INVALID
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] == dataDictionary_in.loc[
                                    row_index + 1, column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_name])):
                                return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if column_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] == dataDictionary_in.loc[
                                    row_index, column_index + 1]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
            else:  # SPECIAL_TYPE is INVALID
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            if column_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] == dataDictionary_in.loc[
                                    row_index, column_index + 1]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_index])):
                                return False

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            if specialTypeInput == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            if idx == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] == dataDictionary_in.loc[idx + 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

                else:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] == dataDictionary_in.loc[idx + 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return False
                            else:
                                if dataDictionary_out.loc[idx, field] == dataDictionary_in.loc[idx + 1, field]:
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False

    return True


def checkDerivedTypeNextNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     specialTypeInput: SpecialType, missing_values: list, axis_param: int, field: str) -> bool:
    """
    Check if the derived type next value is applied correctly when the input and output dataframes when belongOp_in is NOTBELONG and belongOp_out is BELONG
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the next
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the next
    :param specialTypeInput: (SpecialType) Special type to apply the next
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the next
    :param field: (str) Field to apply the next

    :return: True if the derived type next value is applied correctly, False otherwise
    """
    if field is None:
        # Define the lambda function to replace the values within missing values by the value of the next position
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False

            else:  # SPECIAL_TYPE is INVALID
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False
            else:  # SPECIAL_TYPE is INVALID
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            if specialTypeInput == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            return False

                else:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

    return True


def checkDerivedTypeNextNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     specialTypeInput: SpecialType, missing_values: list, axis_param: int, field: str) -> bool:
    """
    Check if the derived type next value is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are NOTBELONG
    :param dataDictionary_in: (pd.DataFrame) Dataframe with the data before the next
    :param dataDictionary_out: (pd.DataFrame) Dataframe with the data after the next
    :param specialTypeInput: (SpecialType) Special type to apply the next
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the next
    :param field: (str) Field to apply the next

    :return: True if the derived type next value is applied correctly, False otherwise
    """
    if field is None:
        # Define the lambda function to replace the values within missing values by the value of the next position
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False

            else:  # SPECIAL_TYPE is INVALID
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False
            else:  # SPECIAL_TYPE is INVALID
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values or pd.isnull(value):
                            return False

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            if specialTypeInput == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            return False

                else:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

    return True


def checkDerivedTypeNext(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
                                  belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                  missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
    """
    Check if the derived type next value is applied correctly
    params:
        :param dataDictionary_in: dataframe with the data before the Next
        :param dataDictionary_out: dataframe with the data after the Next
        :param specialTypeInput: special type to apply the Next
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to apply the Next
        :param field: field to apply the Next

    Returns:
        :return: True if the derived type next value is applied correctly
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result=checkDerivedTypeNextBelongBelong(dataDictionary_in=dataDictionary_in,
                                                    dataDictionary_out=dataDictionary_out,
                                                    specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                    axis_param=axis_param, field=field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result=checkDerivedTypeNextBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                       dataDictionary_out=dataDictionary_out,
                                                       specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                       axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result=checkDerivedTypeNextNotBelongBelong(dataDictionary_in=dataDictionary_in,
                                                       dataDictionary_out=dataDictionary_out,
                                                       specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                       axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result=checkDerivedTypeNextNotBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                          dataDictionary_out=dataDictionary_out,
                                                          specialTypeInput=specialTypeInput,
                                                          missing_values=missing_values, axis_param=axis_param,
                                                          field=field)

    return True if result else False


def checkDerivedTypeColRowOutliers(derivedTypeOutput: DerivedType, dataDictionary_in: pd.DataFrame,
                                   dataDictionary_out: pd.DataFrame, outliers_dataframe_mask: pd.DataFrame,
                                   belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                   axis_param: int = None, field: str = None) -> bool:
    """
    Check the derived type to the outliers of a dataframe
    :param derivedTypeOutput: derived type to apply to the outliers
    :param dataDictionary_in: original dataframe with the data
    :param dataDictionary_out: dataframe with the derived type applied to the outliers
    :param outliers_dataframe_mask: dataframe with the outliers mask
    :param belongOp_in: belong operator condition for the if block of the invariant
    :param belongOp_out: belong operator condition for the else block of the invariant
    :param axis_param: axis to apply the derived type. If axis_param is None, the derived type is applied to the whole dataframe.
    If axis_param is 0, the derived type is applied to each column. If axis_param is 1, the derived type is applied to each row.
    :param field: field to apply the derived type.

    :return: True if the derived type is applied correctly to the outliers, False otherwise
    """
    if field is None:
        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        if np.issubdtype(dataDictionary_in[col].dtype, np.number):
                            for idx, value in dataDictionary_in[col].items():
                                if outliers_dataframe_mask.at[idx, col] == 1:
                                    if dataDictionary_out.at[idx, col] != dataDictionary_in[col].value_counts().idxmax():
                                        return False
                                else:
                                    if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                        return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in[col].value_counts().idxmax():
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False

            elif derivedTypeOutput == DerivedType.PREVIOUS:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != 0:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx - 1, col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != 0:
                                prev_col = row.index[row.index.get_loc(col) - 1]  # Get the previous column
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, prev_col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False

            elif derivedTypeOutput == DerivedType.NEXT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != len(dataDictionary_in) - 1:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx + 1, col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False
                elif axis_param == 1:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != dataDictionary_in.columns[-1]:
                                next_col = dataDictionary_in.columns[dataDictionary_in.columns.get_loc(col) + 1]
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, next_col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False

        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        if np.issubdtype(dataDictionary_in[col].dtype, np.number):
                            for idx, value in dataDictionary_in[col].items():
                                if outliers_dataframe_mask.at[idx, col] == 1:
                                    if dataDictionary_out.at[idx, col] == dataDictionary_in[
                                        col].value_counts().idxmax():
                                        return False
                                else:
                                    if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                        return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1:
                                if dataDictionary_out.at[idx, col] == dataDictionary_in[col].value_counts().idxmax():
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False

            elif derivedTypeOutput == DerivedType.PREVIOUS:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != 0:
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.at[idx - 1, col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != 0:
                                prev_col = row.index[row.index.get_loc(col) - 1]  # Get the previous column
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.at[idx, prev_col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False

            elif derivedTypeOutput == DerivedType.NEXT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != len(dataDictionary_in) - 1:
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.at[idx + 1, col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False
                elif axis_param == 1:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != dataDictionary_in.columns[-1]:
                                next_col = dataDictionary_in.columns[dataDictionary_in.columns.get_loc(col) + 1]
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.at[idx, next_col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]:
                                    return False



        elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        if np.issubdtype(dataDictionary_in[col].dtype, np.number):
                            for idx, value in dataDictionary_in[col].items():
                                if outliers_dataframe_mask.at[idx, col] == 1:
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1:
                                return False

            elif derivedTypeOutput == DerivedType.PREVIOUS:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != 0:
                                return False

                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != 0:
                                return False

            elif derivedTypeOutput == DerivedType.NEXT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != len(dataDictionary_in) - 1:
                                return False
                elif axis_param == 1:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != dataDictionary_in.columns[-1]:
                                return False

        elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        if np.issubdtype(dataDictionary_in[col].dtype, np.number):
                            for idx, value in dataDictionary_in[col].items():
                                if outliers_dataframe_mask.at[idx, col] == 1:
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1:
                                return False

            elif derivedTypeOutput == DerivedType.PREVIOUS:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != 0:
                                return False

                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != 0:
                                return False

            elif derivedTypeOutput == DerivedType.NEXT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != len(dataDictionary_in) - 1:
                                return False
                elif axis_param == 1:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != dataDictionary_in.columns[-1]:
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")
        elif field in outliers_dataframe_mask.columns:
            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in[field].value_counts().idxmax():
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]:
                                return False
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != 0:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx - 1, field]:
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]:
                                return False
                elif derivedTypeOutput == DerivedType.NEXT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != len(dataDictionary_in) - 1:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx + 1, field]:
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]:
                                return False

            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1:
                            if dataDictionary_out.at[idx, field] == dataDictionary_in[field].value_counts().idxmax():
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]:
                                return False
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != 0:
                            if dataDictionary_out.at[idx, field] == dataDictionary_in.at[idx - 1, field]:
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]:
                                return False
                elif derivedTypeOutput == DerivedType.NEXT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != len(dataDictionary_in) - 1:
                            if dataDictionary_out.at[idx, field] == dataDictionary_in.at[idx + 1, field]:
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]:
                                return False

            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1:
                            return False
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != 0:
                            return False
                elif derivedTypeOutput == DerivedType.NEXT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != len(dataDictionary_in) - 1:
                            return False

            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1:
                            return False
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != 0:
                            return False
                elif derivedTypeOutput == DerivedType.NEXT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != len(dataDictionary_in) - 1:
                            return False

    return True
