# Importing enumerations from packages
from typing import Union
from helpers.enumerations import Operator, DataType, SpecialType, DerivedType, Belong

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


def check_derivedType(specialTypeInput: SpecialType, derivedTypeOutput: DerivedType,
                      dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                      belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                      missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
    """
    Check the derived type to the missing values of a dataframe
    :param specialTypeInput: special type to apply to the missing values
    :param derivedTypeOutput: derived type to apply to the missing values
    :param dataDictionary_in: original dataframe with the data
    :param dataDictionary_out: dataframe with the derived type applied to the missing values
    :param belongOp_in: belong operator condition for the if block of the invariant
    :param belongOp_out: belong operator condition for the else block of the invariant
    :param missing_values: list of missing values
    :param axis_param: axis to apply the derived type. If axis_param is None, the derived type is applied to the whole dataframe.
    If axis_param is 0, the derived type is applied to each column. If axis_param is 1, the derived type is applied to each row.
    :param field: field to apply the derived type.

    :return: True if the derived type is correctly applied to the missing values, False otherwise
    """
    result = True
    if field is None:
        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
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
                                    if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[row_index, column_name]:
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
                                    if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[row_index, column_name] and not(pd.isnull(dataDictionary_in.loc[row_index, column_name]) and pd.isnull(dataDictionary_out.loc[row_index, column_name])):
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
                                    if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[row_index, column_index]:
                                        return False
                    if specialTypeInput == SpecialType.INVALID:
                        for row_index, row in dataDictionary_in.iterrows():
                            most_frequent = row.value_counts().idxmax()
                            for column_index, value in row.items():
                                if value in missing_values:
                                    if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                        return False
                                else:
                                    if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[row_index, column_index] and not(pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(dataDictionary_out.loc[row_index, column_index])):
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
                                    if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[row_index, column_index]:
                                        return False
                    if specialTypeInput == SpecialType.INVALID:
                        for row_index, row in dataDictionary_in.iterrows():
                            for column_index, value in row.items():
                                if value in missing_values:
                                    if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                        return False
                                else:
                                    if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[row_index, column_index] and not(pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(dataDictionary_out.loc[row_index, column_index])):
                                        return False

            elif derivedTypeOutput == DerivedType.PREVIOUS:
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
                                        if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[row_index - 1, column_name]:
                                            return False
                                else:
                                    if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                        row_index, column_name]:
                                        return False

                    else: # SPECIAL_TYPE is INVALID
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

            elif derivedTypeOutput == DerivedType.NEXT:
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
                                        if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[row_index + 1, column_name]:
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
                                        if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[row_index + 1, column_name]:
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
                                        if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[row_index, column_index+1]:
                                            return False
                                else:
                                    if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                        row_index, column_index]:
                                        return False
                    else: # SPECIAL_TYPE is INVALID
                        for row_index, row in dataDictionary_in.iterrows():
                            for column_index, value in row.items():
                                if value in missing_values or pd.isnull(value):
                                    if column_index == len(dataDictionary_in) - 1:
                                        if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                          row_index, column_index]:
                                            return False
                                    else:
                                        if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[row_index, column_index+1]:
                                            return False
                                else:
                                    if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                        row_index, column_index] and not (pd.isnull(
                                        dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                        dataDictionary_out.loc[row_index, column_index])):
                                        return False

                elif axis_param is None:
                    raise ValueError("The axis cannot be None when applying the NEXT operation")

        if belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
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

            elif derivedTypeOutput == DerivedType.PREVIOUS:
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

            elif derivedTypeOutput == DerivedType.NEXT:
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
            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
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
                elif derivedTypeOutput == DerivedType.PREVIOUS:
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

                elif derivedTypeOutput == DerivedType.NEXT:
                    if specialTypeInput == SpecialType.MISSING:
                        if missing_values is not None:
                            # result = (pd.Series([dataDictionary_in[field].iloc[i + 1]
                            #                     if (value in missing_values or pd.isnull(value)) and i < len(dataDictionary_in[field]) - 1
                            #                     else value for i, value in enumerate(dataDictionary_in[field])],
                            #                    index=dataDictionary_in[field].index)
                            #           .equals(dataDictionary_out[field]))
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

            if belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
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
                elif derivedTypeOutput == DerivedType.PREVIOUS:
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

                elif derivedTypeOutput == DerivedType.NEXT:
                    if specialTypeInput == SpecialType.MISSING:
                        if missing_values is not None:
                            # result = (pd.Series([dataDictionary_in[field].iloc[i + 1]
                            #                     if (value in missing_values or pd.isnull(value)) and i < len(dataDictionary_in[field]) - 1
                            #                     else value for i, value in enumerate(dataDictionary_in[field])],
                            #                    index=dataDictionary_in[field].index)
                            #           .equals(dataDictionary_out[field]))
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

            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
                if specialTypeInput == SpecialType.MISSING:
                    for row_index, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            return False
                elif specialTypeInput == SpecialType.INVALID:
                    for row_index, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
                if specialTypeInput == SpecialType.MISSING:
                    for row_index, value in dataDictionary_in[field].items():
                        if value in missing_values or pd.isnull(value):
                            return False
                elif specialTypeInput == SpecialType.INVALID:
                    for row_index, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            return False

    return result


def check_derivedTypeColRowOutliers(derivedTypeOutput: DerivedType, dataDictionary_in: pd.DataFrame,
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
