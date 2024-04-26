# Importing libraries
import numpy as np
import pandas as pd

# Importing enumerations from packages
from helpers.auxiliar import find_closest_value, check_interval_condition
from helpers.enumerations import DerivedType, Belong, SpecialType, Closure


def checkFixValueMostFrequentBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                    fixValueInput, belongOp_out: Belong, axis_param: int, field: str) -> bool:
    """
        Check if the most frequent value is applied correctly on the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is always BELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
            :param fixValueInput: input value to apply the most frequent value
            :param belongOp_out: (Belong) then condition to check the invariant
            :param axis_param: (int) axis to apply the most frequent value
            :param field: (str) field to apply the most frequent value

        Returns:
            :return: True if the most frequent value is applied correctly on the fix input value
        """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                most_frequent_value = row.value_counts().idxmax()
                for column_index, value in row.items():
                    if value == fixValueInput:
                        if dataDictionary_out.at[row_index, column_index] != most_frequent_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                            row_index, column_index]:
                            return False
        elif axis_param == 0:  # Applies the lambda function at the column level
            for col in dataDictionary_in.columns:
                most_frequent_value = dataDictionary_in[col].value_counts().idxmax()
                for idx, value in dataDictionary_in[col].items():
                    if value == fixValueInput:
                        if dataDictionary_out.at[idx, col] != most_frequent_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col]:
                            return False
        else:  # Applies at the dataframe level
            # Calculate the most frequent value
            most_frequent_value = dataDictionary_in.stack().value_counts().idxmax()
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if value == fixValueInput:
                        if dataDictionary_out.at[idx, col_name] != most_frequent_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            most_frequent_value = dataDictionary_in[field].value_counts().idxmax()
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    if dataDictionary_out.at[idx, field] != most_frequent_value:
                        if belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_out == Belong.NOTBELONG:
                            return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False


def checkFixValueMostFrequentNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                             fixValueInput, axis_param: int, field: str) -> bool:
    """
        Check if the most frequent value is applied correctly to the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is NOTBELONG and belongOp_out is BELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
            :param fixValueInput: input value to apply the most frequent value
            :param axis_param: (int) axis to apply the most frequent value
            :param field: (str) field to apply the most frequent value

        Returns:
            :return: True if the most frequent value is applied correctly to the fix input value
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_index, value in row.items():
                    if value == fixValueInput:
                        return False
        elif axis_param == 0:  # Applies the lambda function at the column level
            for col in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col].items():
                    if value == fixValueInput:
                        return False
        else:  # Applies at the dataframe level
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if value == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    return False

    return True


def checkFixValueMostFrequentNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                                fixValueInput, axis_param: int, field: str) -> bool:
    """
        Check if the most frequent value is applied correctly to the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in and belongOp_out are NOTBELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
            :param fixValueInput: input value to apply the most frequent value
            :param axis_param: (int) axis to apply the most frequent value
            :param field: (str) field to apply the most frequent value

        Returns:
            :return: True if the most frequent value is applied correctly to the fix input value
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_index, value in row.items():
                    if value == fixValueInput:
                        return False
        elif axis_param == 0:  # Applies the lambda function at the column level
            for col in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col].items():
                    if value == fixValueInput:
                        return False
        else:  # Applies at the dataframe level
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if value == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    return False

    return True


def checkFixValueMostFrequent(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                              fixValueInput, belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                              axis_param: int = None, field: str = None) -> bool:
    """
        Check if the most frequent value is applied correctly on the fix input value
        to the dataDictionary_out respect to the dataDictionary_in

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
            :param fixValueInput: input value to apply the most frequent value
            :param belongOp_in: (Belong) if condition to check the invariant
            :param belongOp_out: (Belong) then condition to check the invariant
            :param axis_param: (int) axis to apply the most frequent value
            :param field: (str) field to apply the most frequent value

        Returns:
            :return: True if the most frequent value is applied correctly on the fix input value
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result = checkFixValueMostFrequentBelong(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_out, axis_param, field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValueMostFrequentBelong(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_out, axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkFixValueMostFrequentNotBelongBelong(dataDictionary_in, dataDictionary_out, fixValueInput, axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValueMostFrequentNotBelongNotBelong(dataDictionary_in, dataDictionary_out, fixValueInput, axis_param, field)

    return True if result else False


def checkFixValuePreviousBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                                 belongOp_out: Belong, axis_param: int, field: str) -> bool:
    """
        Check if the previous value is applied correctly to the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is always BELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the previous value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the previous value
            :param fixValueInput: input value to apply the previous value
            :param belongOp_out: (Belong) then condition to check the invariant
            :param axis_param: (int) axis to apply the previous value
            :param field: (str) field to apply the previous value

        Returns:
            :return: True if the previous value is applied correctly to the fix input value
        """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    column_index = dataDictionary_in.columns.get_loc(column_name)
                    value = dataDictionary_in.at[row_index, column_name]
                    if value == fixValueInput:
                        if column_index == 0:
                            if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                                row_index, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if column_index - 1 in dataDictionary_in.columns:
                                if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                                    row_index, column_index - 1]:
                                    if belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_out == Belong.NOTBELONG:
                                        return True
                    else:
                        if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                            row_index, column_index]:
                            return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if value == fixValueInput:
                        if row_index == 0:
                            if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                                row_index, column_name] or (
                                    not pd.isnull(dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index - 1, column_name])):
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index - 1, column_name] or (
                                    not pd.isnull(dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index - 1, column_name])):
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                    else:
                        if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                            row_index, column_name]:
                            return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    if idx == 0:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False


def checkFixValuePreviousNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                         fixValueInput, axis_param: int, field: str) -> bool:
    """
        Check if the previous value is applied correctly to the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is NOTBELONG and belongOp_out is BELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the previous value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the previous value
            :param fixValueInput: input value to apply the previous value
            :param axis_param: (int) axis to apply the previous value
            :param field: (str) field to apply the previous value

        Returns:
            :return: True if the previous value is applied correctly to the fix input value
        """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    value = dataDictionary_in.at[row_index, column_name]
                    if value == fixValueInput:
                        return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if value == fixValueInput:
                        return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    return False

    return True


def checkFixValuePreviousNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                            fixValueInput, axis_param: int, field: str) -> bool:
    """
        Check if the previous value is applied correctly to the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in and belongOp_out are NOTBELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the previous value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the previous value
            :param fixValueInput: input value to apply the previous value
            :param axis_param: (int) axis to apply the previous value
            :param field: (str) field to apply the previous value
            :param axis_param: (int) axis to apply the previous value
            :param field: (str) field to apply the previous value

        Returns:
            :return: True if the previous value is applied correctly to the fix input value
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    value = dataDictionary_in.at[row_index, column_name]
                    if value == fixValueInput:
                        return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if value == fixValueInput:
                        return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    return False

    return True


def checkFixValuePrevious(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                          belongOp_in: Belong, belongOp_out: Belong,
                          axis_param: int, field: str) -> bool:
    """
        Check if the previous value is applied correctly on the fix input value
        to the dataDictionary_out respect to the dataDictionary_in

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the previous value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the previous value
            :param fixValueInput: input value to apply the previous value
            :param belongOp_in: (Belong) if condition to check the invariant
            :param belongOp_out: (Belong) then condition to check the invariant
            :param axis_param: (int) axis to apply the previous value
            :param field: (str) field to apply the previous value

        Returns:
            :return: True if the previous value is applied correctly on the fix input value
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result = checkFixValuePreviousBelong(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_out,
                                                 axis_param, field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValuePreviousBelong(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_out,
                                                 axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkFixValuePreviousNotBelongBelong(dataDictionary_in, dataDictionary_out, fixValueInput,
                                                          axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValuePreviousNotBelongNotBelong(dataDictionary_in, dataDictionary_out, fixValueInput,
                                                             axis_param, field)

    return True if result else False


def checkFixValueNextBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                            belongOp_out: Belong, axis_param: int, field: str) -> bool:
    """
        Check if the next value is applied correctly to the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is always BELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the next value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the next value
            :param fixValueInput: input value to apply the next value
            :param belongOp_out: (Belong) then condition to check the invariant
            :param axis_param: (int) axis to apply the next value
            :param field: (str) field to apply the next value

        Returns:
            :return: True if the next value is applied correctly to the fix input value
        """
    if field is None:
        # TODO: fix this case when axis_param=1 for every invariant auxiliar functions
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    column_index = dataDictionary_in.columns.get_loc(column_name)
                    value = dataDictionary_in.at[row_index, column_name]
                    if value == fixValueInput:
                        if column_index == len(row) - 1:
                            if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                                row_index, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.iloc[
                                row_index, column_index + 1]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                    else:
                        if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                            row_index, column_name]:
                            return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if value == fixValueInput:
                        if row_index == len(dataDictionary_in) - 1:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index + 1, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                    else:
                        if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                            row_index, column_name]:
                            return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    if idx == len(dataDictionary_in) - 1:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx + 1, field]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False


def checkFixValueNextNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     fixValueInput, axis_param: int, field: str) -> bool:
    """
        Check if the next value is applied correctly to the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is NOTBELONG and belongOp_out is BELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the next value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the next value
            :param fixValueInput: input value to apply the next value
            :param axis_param: (int) axis to apply the next value
            :param field: (str) field to apply the next value

        Returns:
            :return: True if the next value is applied correctly to the fix input value
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    value = dataDictionary_in.at[row_index, column_name]
                    if value == fixValueInput:
                        return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if value == fixValueInput:
                        return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    return False

    return True


def checkFixValueNextNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        fixValueInput, axis_param: int, field: str) -> bool:
    """
        Check if the next value is applied correctly to the fix input value
        to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is NOTBELONG and belongOp_out is NOTBELONG

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the next value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the next value
            :param fixValueInput: input value to apply the next value
            :param axis_param: (int) axis to apply the next value
            :param field: (str) field to apply the next value

        Returns:
            :return: True if the next value is applied correctly to the fix input value
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    value = dataDictionary_in.at[row_index, column_name]
                    if value == fixValueInput:
                        return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if value == fixValueInput:
                        return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if value == fixValueInput:
                    return False

    return True



def checkFixValueNext(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                      belongOp_in: Belong, belongOp_out: Belong,
                      axis_param: int, field: str) -> bool:
    """
        Check if the next value is applied correctly on the fix input value
        to the dataDictionary_out respect to the dataDictionary_in

        params:
            :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the next value
            :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the next value
            :param fixValueInput: input value to apply the next value
            :param belongOp_in: (Belong) if condition to check the invariant
            :param belongOp_out: (Belong) then condition to check the invariant
            :param axis_param: (int) axis to apply the next value
            :param field: (str) field to apply the next value

        Returns:
            :return: True if the next value is applied correctly on the fix input value
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result = checkFixValueNextBelong(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_out,
                                                 axis_param, field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValueNextBelong(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_out,
                                                 axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkFixValueNextNotBelongBelong(dataDictionary_in, dataDictionary_out, fixValueInput,
                                                          axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValueNextNotBelongNotBelong(dataDictionary_in, dataDictionary_out, fixValueInput,
                                                             axis_param, field)

    return True if result else False



def checkIntervalMostFrequentBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                    leftMargin: float, rightMargin: float, closureType: Closure,
                                    belongOp_out: Belong,
                                    axis_param: int = None, field: str = None) -> bool:
    """
    Check if the most frequent value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is always BELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the most frequent value
        :param field: (str) field to apply the most frequent value

    Returns:
        :return: True if the most frequent value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                most_frequent_value = row.value_counts().idxmax()
                for column_index, value in row.items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        if dataDictionary_out.at[row_index, column_index] != most_frequent_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[row_index, column_index]:
                            return False
        elif axis_param == 0:  # Applies the lambda function at the column level
            for col in dataDictionary_in.columns:
                most_frequent_value = dataDictionary_in[col].value_counts().idxmax()
                for idx, value in dataDictionary_in[col].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        if dataDictionary_out.at[idx, col] != most_frequent_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col]:
                            return False
        else:  # Applies at the dataframe level
            # Calculate the most frequent value
            most_frequent_value = dataDictionary_in.stack().value_counts().idxmax()
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        if dataDictionary_out.at[idx, col_name] != most_frequent_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            most_frequent_value = dataDictionary_in[field].value_counts().idxmax()
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    if dataDictionary_out.at[idx, field] != most_frequent_value:
                        if belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_out == Belong.NOTBELONG:
                            return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False


def checkIntervalMostFrequentNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                             leftMargin: float, rightMargin: float, closureType: Closure,
                                             axis_param: int = None, field: str = None) -> bool:
    """
    Check if the most frequent value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is NOTBELONG and belongOp_out is BELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param axis_param: (int) axis to apply the most frequent value
        :param field: (str) field to apply the most frequent value

    Returns:
        :return: True if the most frequent value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_index, value in row.items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        elif axis_param == 0:  # Applies the lambda function at the column level
            for col in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        else:  # Applies at the dataframe level
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    return False

    return True


def checkIntervalMostFrequentNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                                leftMargin: float, rightMargin: float, closureType: Closure,
                                                axis_param: int = None, field: str = None) -> bool:
    """
    Check if the most frequent value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in and belongOp_out are NOTBELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param axis_param: (int) axis to apply the most frequent value
        :param field: (str) field to apply the most frequent value

    Returns:
        :return: True if the most frequent value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_index, value in row.items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        elif axis_param == 0:  # Applies the lambda function at the column level
            for col in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        else:  # Applies at the dataframe level
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    return False

    return True


def checkIntervalMostFrequent(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                               leftMargin: float, rightMargin: float, closureType: Closure,
                               belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                               axis_param: int = None, field: str = None) -> bool:
    """
    Check if the most frequent value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the most frequent value
        :param field: (str) field to apply the most frequent value

    Returns:
        :return: True if the most frequent value is applied correctly on the interval
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalMostFrequentBelong(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                                 closureType, belongOp_out, axis_param, field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalMostFrequentBelong(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                                 closureType, belongOp_out, axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalMostFrequentNotBelongBelong(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                                          closureType, axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalMostFrequentNotBelongNotBelong(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                                             closureType, axis_param, field)

    return True if result else False


def checkIntervalPreviousBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                leftMargin: float, rightMargin: float, closureType: Closure,
                                belongOp_out: Belong = Belong.BELONG,
                                axis_param: int = None, field: str = None) -> bool:
    """
    Check if the previous value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is always BELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the previous value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the previous value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the previous value
        :param field: (str) field to apply the previous value

    Returns:
        :return: True if the previous value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    column_index = dataDictionary_in.columns.get_loc(column_name)
                    value = dataDictionary_in.at[row_index, column_name]
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        if column_index == 0:
                            if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                                row_index, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if column_index - 1 in dataDictionary_in.columns:
                                if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                                    row_index, column_index - 1]:
                                    if belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_out == Belong.NOTBELONG:
                                        return True
                    else:
                        if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                            row_index, column_index]:
                            return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        if row_index == 0:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index - 1, column_name] and (
                                    not pd.isnull(dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index - 1, column_name])):
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                    else:
                        if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                            row_index, column_name]:
                            return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    if idx == 0:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False


def checkIntervalPreviousNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                             leftMargin: float, rightMargin: float, closureType: Closure,
                                             axis_param: int = None, field: str = None) -> bool:
    """
    Check if the previous value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is NOTBELONG and belongOp_out is BELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the previous value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the previous value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param axis_param: (int) axis to apply the previous value
        :param field: (str) field to apply the previous value

    Returns:
        :return: True if the previous value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    value = dataDictionary_in.at[row_index, column_name]
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    return False

    return True


def checkIntervalPreviousNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                             leftMargin: float, rightMargin: float, closureType: Closure,
                                             axis_param: int = None, field: str = None) -> bool:
    """
    Check if the previous value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in and belongOp_out are NOTBELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the previous value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the previous value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param axis_param: (int) axis to apply the previous value
        :param field: (str) field to apply the previous value

    Returns:
        :return: True if the previous value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    value = dataDictionary_in.at[row_index, column_name]
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    return False

    return True


def checkIntervalPrevious(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                           leftMargin: float, rightMargin: float, closureType: Closure,
                           belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                           axis_param: int = None, field: str = None) -> bool:
    """
    Check if the previous value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the previous value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the previous value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the previous value
        :param field: (str) field to apply the previous value

    Returns:
        :return: True if the previous value is applied correctly on the interval
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalPreviousBelong(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                                 closureType, belongOp_out, axis_param, field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalPreviousBelong(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                                 closureType, belongOp_out, axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalPreviousNotBelongBelong(dataDictionary_in, dataDictionary_out, leftMargin,
                                                          rightMargin,
                                                          closureType, axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalPreviousNotBelongNotBelong(dataDictionary_in, dataDictionary_out, leftMargin,
                                                             rightMargin,
                                                             closureType, axis_param, field)

    return True if result else False


def checkIntervalNextBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_out: Belong = Belong.BELONG,
                            axis_param: int = None, field: str = None) -> bool:
    """
    Check if the next value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is always BELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the next value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the next value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the next value
        :param field: (str) field to apply the next value

    Returns:
        :return: True if the next value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    column_index = dataDictionary_in.columns.get_loc(column_name)
                    value = dataDictionary_in.at[row_index, column_name]
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        if column_index == len(row) - 1:
                            if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                                row_index, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.iloc[
                                row_index, column_index + 1]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                    else:
                        if dataDictionary_out.at[row_index, column_name] != dataDictionary_in.at[
                            row_index, column_name]:
                            return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        if row_index == len(dataDictionary_in) - 1:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index + 1, column_name]:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                    else:
                        if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                            row_index, column_name]:
                            return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    if idx == len(dataDictionary_in) - 1:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx + 1, field]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False


def checkIntervalNextNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     leftMargin: float, rightMargin: float, closureType: Closure,
                                     axis_param: int = None, field: str = None) -> bool:
    """
    Check if the next value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in is NOTBELONG and belongOp_out is BELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the next value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the next value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param axis_param: (int) axis to apply the next value
        :param field: (str) field to apply the next value

    Returns:
        :return: True if the next value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    value = dataDictionary_in.at[row_index, column_name]
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    return False

    return True


def checkIntervalNextNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                         leftMargin: float, rightMargin: float, closureType: Closure,
                                         axis_param: int = None, field: str = None) -> bool:
    """
    Check if the next value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in when belongOp_in and belongOp_out are NOTBELONG

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the next value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the next value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param axis_param: (int) axis to apply the next value
        :param field: (str) field to apply the next value

    Returns:
        :return: True if the next value is applied correctly on the interval
    """
    if field is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in dataDictionary_in.iterrows():
                for column_name, value in row.items():
                    value = dataDictionary_in.at[row_index, column_name]
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field in dataDictionary_in.columns:
            for idx, value in dataDictionary_in[field].items():
                if check_interval_condition(value, leftMargin, rightMargin, closureType):
                    return False

    return True


def checkIntervalNext(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                       leftMargin: float, rightMargin: float, closureType: Closure,
                       belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                       axis_param: int = None, field: str = None) -> bool:
    """
    Check if the next value is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the next value
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the next value
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the next value
        :param field: (str) field to apply the next value

    Returns:
        :return: True if the next value is applied correctly on the interval
    """
    result = True

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalNextBelong(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                             closureType, belongOp_out, axis_param, field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalNextBelong(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                             closureType, belongOp_out, axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalNextNotBelongBelong(dataDictionary_in, dataDictionary_out, leftMargin,
                                                      rightMargin,
                                                      closureType, axis_param, field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalNextNotBelongNotBelong(dataDictionary_in, dataDictionary_out, leftMargin,
                                                         rightMargin,
                                                         closureType, axis_param, field)

    return True if result else False


def checkFixValueInterpolationBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     fixValueInput, belongOp_out: Belong, axis_param: int = None,
                                     field: str = None) -> bool:
    """
        Check if the interpolation is applied correctly to the fix value input when the input and output dataframe
        when belongOp_in is BELONG and belongOp_out is BELONG or NOTBELONG
        params:
            :param dataDictionary_in: dataframe with the data before the interpolation
            :param dataDictionary_out: dataframe with the data after the interpolation
            :param fixValueInput: value to apply the interpolation
            :param belongOp_out: then condition to check the invariant
            :param axis_param: axis to apply the interpolation
            :param field: field to apply the interpolation

        Returns:
            :return: True if the interpolation is applied correctly to the fix value input
    """
    dataDictionary_in_copy = dataDictionary_in.copy()
    if field is None:
        if axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                dataDictionary_in_copy[col_name] = (
                    dataDictionary_in[col_name].apply(lambda x: np.nan if x == fixValueInput else x).
                    interpolate(method='linear', limit_direction='both'))

            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                for idx in dataDictionary_in.index:
                    if dataDictionary_in.at[idx, col_name] == fixValueInput:
                        if dataDictionary_out.at[idx, col_name] != dataDictionary_in_copy.at[idx, col_name]:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if (dataDictionary_out.at[idx, col_name] != dataDictionary_in.at[idx, col_name]) and not(pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(dataDictionary_out.at[idx, col_name])):
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                dataDictionary_in_copy[row] = (
                    numeric_data[row].apply(lambda x: np.nan if x == fixValueInput else x).
                    interpolate(method='linear', limit_direction='both'))
            for col_name in dataDictionary_in.columns:
                for idx in dataDictionary_in.index:
                    if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if (dataDictionary_out.at[idx, col_name] != dataDictionary_in.at[idx, col_name]) and not(pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(dataDictionary_out.at[idx, col_name])):
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        dataDictionary_in_copy[field] = (dataDictionary_in[field].apply(lambda x: np.nan if x == fixValueInput else x).
                                         interpolate(method='linear', limit_direction='both'))

        for idx in dataDictionary_in.index:
            if dataDictionary_in.at[idx, field] == fixValueInput:
                if dataDictionary_out.at[idx, field] != dataDictionary_in_copy.at[idx, field]:
                    if belongOp_out == Belong.BELONG:
                        return False
                    elif belongOp_out == Belong.NOTBELONG:
                        return True
            else:
                if (dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]) and not(pd.isnull(dataDictionary_out.at[idx, field]) or pd.isnull(dataDictionary_out.at[idx, field])):
                    return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkFixValueInterpolationNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                              fixValueInput, axis_param: int, field: str) -> bool:
    """
        Check if the interpolation is applied correctly to the fix value input when the input and output dataframe
        when belongOp_in is NOTBELONG and belongOp_out is BELONG
        params:
            :param dataDictionary_in: dataframe with the data before the interpolation
            :param dataDictionary_out: dataframe with the data after the interpolation
            :param fixValueInput: value to apply the interpolation
            :param axis_param: axis to apply the interpolation
            :param field: field to apply the interpolation

        Returns:
            :return: True if the interpolation is applied correctly to the fix value input
    """
    dataDictionary_in_copy = dataDictionary_in.copy()
    if field is None:
        if axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                dataDictionary_in_copy[col_name] = (
                    dataDictionary_in[col_name].apply(lambda x: np.nan if x == fixValueInput else x).
                    interpolate(method='linear', limit_direction='both'))

            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                for idx in dataDictionary_in.index:
                    if dataDictionary_in.at[idx, col_name] == fixValueInput:
                       return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                dataDictionary_in_copy[row] = (
                    numeric_data[row].apply(lambda x: np.nan if x == fixValueInput else x).
                    interpolate(method='linear', limit_direction='both'))
            for col_name in dataDictionary_in.columns:
                for idx in dataDictionary_in.index:
                    if dataDictionary_in.at[idx, col_name] == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        dataDictionary_in_copy[field] = (dataDictionary_in[field].apply(lambda x: np.nan if x == fixValueInput else x).
                                         interpolate(method='linear', limit_direction='both'))

        for idx in dataDictionary_in.index:
            if dataDictionary_in.at[idx, field] == fixValueInput:
                return False

    return True


def checkFixValueInterpolationNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                                 fixValueInput, axis_param: int, field: str) -> bool:
    """
        Check if the interpolation is applied correctly to the fix value input when the input and output dataframe
        when belongOp_in is NOTBELONG and belongOp_out is NOTBELONG
        params:
            :param dataDictionary_in: dataframe with the data before the interpolation
            :param dataDictionary_out: dataframe with the data after the interpolation
            :param fixValueInput: value to apply the interpolation
            :param axis_param: axis to apply the interpolation
            :param field: field to apply the interpolation

        Returns:
            :return: True if the interpolation is applied correctly to the fix value input
    """
    dataDictionary_in_copy = dataDictionary_in.copy()
    if field is None:
        if axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                dataDictionary_in_copy[col_name] = (
                    dataDictionary_in[col_name].apply(lambda x: np.nan if x == fixValueInput else x).
                    interpolate(method='linear', limit_direction='both'))

            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                for idx in dataDictionary_in.index:
                    if dataDictionary_in.at[idx, col_name] == fixValueInput:
                        return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                dataDictionary_in_copy[row] = (
                    numeric_data[row].apply(lambda x: np.nan if x == fixValueInput else x).
                    interpolate(method='linear', limit_direction='both'))
            for col_name in dataDictionary_in.columns:
                for idx in dataDictionary_in.index:
                    if dataDictionary_in.at[idx, col_name] == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        dataDictionary_in_copy[field] = (dataDictionary_in[field].apply(lambda x: np.nan if x == fixValueInput else x).
                                         interpolate(method='linear', limit_direction='both'))

        for idx in dataDictionary_in.index:
            if dataDictionary_in.at[idx, field] == fixValueInput:
                return False

    return True


def checkFixValueInterpolation(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                               fixValueInput, belongOp_in: Belong = Belong.BELONG,
                               belongOp_out: Belong = Belong.BELONG, axis_param: int = None,
                               field: str = None) -> bool:
    """
        Check if the interpolation is applied correctly to the fix value in the dataDictionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: dataframe with the data before the interpolation
            :param dataDictionary_out: dataframe with the data after the interpolation
            :param fixValueInput: fix value to apply the interpolation
            :param belongOp_in: if condition to check the invariant
            :param belongOp_out: then condition to check the invariant
            :param axis_param: axis to apply the interpolation
            :param field: field to apply the interpolation

        Returns:
            :return: True if the interpolation is applied correctly to the fix value
    """
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (
            belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkFixValueInterpolationBelong(dataDictionary_in=dataDictionary_in,
                                                     dataDictionary_out=dataDictionary_out,
                                                     fixValueInput=fixValueInput,
                                                     belongOp_out=belongOp_out,
                                                     axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkFixValueInterpolationNotBelongBelong(dataDictionary_in=dataDictionary_in,
                                                           dataDictionary_out=dataDictionary_out,
                                                           fixValueInput=fixValueInput,
                                                            axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValueInterpolationNotBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                                 dataDictionary_out=dataDictionary_out,
                                                                 fixValueInput=fixValueInput,
                                                                 axis_param=axis_param, field=field)

    return True if result else False


def checkFixValueMeanBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                            belongOp_out: Belong, axis_param: int = None, field: str = None) -> bool:
    """
        Check if the special type mean is applied correctly when the input and output dataframes
        when belongOp_in and belongOp_out are BELONG
        params::
            :param dataDictionary_in: dataframe with the data before the mean
            :param dataDictionary_out: dataframe with the data after the mean
            :param fixValueInput: fix value to apply the mean
            :param belongOp_out: then condition to check the invariant
            :param axis_param: axis to apply the mean
            :param field: field to apply the mean

        Returns:
            :return: True if the special type mean is applied correctly
        """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the mean of these numeric columns
            mean_value = only_numbers_df.mean().mean()
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            if dataDictionary_out.at[idx, col_name] != mean_value:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                    dataDictionary_out.at[idx, col_name])):
                                return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                mean = dataDictionary_in[col_name].mean()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            if dataDictionary_out.at[idx, col_name] != mean:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                    dataDictionary_out.at[idx, col_name])):
                                return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                mean = numeric_data.mean()
                # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if value == fixValueInput:
                        if dataDictionary_out.at[idx, col_name] != mean:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                dataDictionary_out.at[idx, col_name])):
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the mean
        mean = dataDictionary_in[field].mean()
        for idx, value in dataDictionary_in[field].items():
            if value == fixValueInput:
                if dataDictionary_out.at[idx, field] != mean:
                    if belongOp_out == Belong.BELONG:
                        return False
                    elif belongOp_out == Belong.NOTBELONG:
                        return True
            else:
                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                    return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkFixValueMeanNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                                     axis_param: int, field: str) -> bool:
    """
        Check if the special type mean is applied correctly when the input and output dataframes
        when belongOp_in is NOTBELONG and belongOp_out is BELONG
        params::
            :param dataDictionary_in: dataframe with the data before the mean
            :param dataDictionary_out: dataframe with the data after the mean
            :param fixValueInput: fix value to apply the mean
            :param axis_param: axis to apply the mean
            :param field: field to apply the mean

        Returns:
            :return: True if the special type mean is applied correctly
        """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the mean of these numeric columns
            mean_value = only_numbers_df.mean().mean()
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                mean = dataDictionary_in[col_name].mean()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                mean = numeric_data.mean()
                # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if value == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the mean
        mean = dataDictionary_in[field].mean()
        for idx, value in dataDictionary_in[field].items():
            if value == fixValueInput:
                return False

        return True


def checkFixValueMeanNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                                        axis_param: int, field: str) -> bool:
    """
        Check if the special type mean is applied correctly when the input and output dataframes
        when belongOp_in is and belongOp_out are NOTBELONG
        params::
            :param dataDictionary_in: dataframe with the data before the mean
            :param dataDictionary_out: dataframe with the data after the mean
            :param fixValueInput: fix value to apply the mean
            :param axis_param: axis to apply the mean
            :param field: field to apply the mean

        Returns:
            :return: True if the special type mean is applied correctly
        """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the mean of these numeric columns
            mean_value = only_numbers_df.mean().mean()
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                mean = dataDictionary_in[col_name].mean()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                mean = numeric_data.mean()
                # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if value == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the mean
        mean = dataDictionary_in[field].mean()
        for idx, value in dataDictionary_in[field].items():
            if value == fixValueInput:
                return False

        return True


def checkFixValueMean(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                      belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                      axis_param: int = None, field: str = None) -> bool:
    """
    Check if the mean is applied correctly to the fix value in the dataDictionary_out
    respect to the dataDictionary_in
    params:
        :param dataDictionary_in: dataframe with the data before the mean
        :param dataDictionary_out: dataframe with the data after the mean
        :param fixValueInput: fix value to apply the mean
        :param belongOp_in: if condition to check the mean
        :param belongOp_out: then condition to check the mean
        :param axis_param: axis to apply the mean
        :param field: field to apply the mean

    Returns:
        :return: True if the mean is applied correctly to the fix value
    """
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (
            belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkFixValueMeanBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                            fixValueInput=fixValueInput, belongOp_out=belongOp_out,
                                            axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkFixValueMeanNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                     fixValueInput=fixValueInput, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValueMeanNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        fixValueInput=fixValueInput, axis_param=axis_param, field=field)

    return True if result else False


def checkFixValueMedianBelong(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_out, axis_param, field):
    """
        Check if the median is applied correctly to the fix input value when the input and output dataframes
        when belongOp_in and belongOp_out are BELONG
        params::
            :param dataDictionary_in: dataframe with the data before the median
            :param dataDictionary_out: dataframe with the data after the median
            :param fixValueInput: fix value to apply the median
            :param belongOp_out: then condition to check the invariant
            :param axis_param: axis to apply the median
            :param field: field to apply the median

        Returns:
            :return: True if the median is applied correctly to the fix value
        """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the median of these numeric columns
            median_value = only_numbers_df.median().median()
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            if dataDictionary_out.at[idx, col_name] != median_value:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
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
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            if dataDictionary_out.at[idx, col_name] != median:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                median = numeric_data.median()
                # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if value == fixValueInput:
                        if dataDictionary_out.at[idx, col_name] != median:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        # Check the dataDictionary_out positions with missing values have been replaced with the median
        median = dataDictionary_in[field].median()
        for idx, value in dataDictionary_in[field].items():
            if value == fixValueInput:
                if dataDictionary_out.at[idx, field] != median:
                    if belongOp_out == Belong.BELONG:
                        return False
                    elif belongOp_out == Belong.NOTBELONG:
                        return True
            else:
                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                    return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkFixValueMedianNotBelongBelong(dataDictionary_in, dataDictionary_out, fixValueInput, axis_param, field):
    """
            Check if the median is applied correctly to the fix input value when the input and output dataframes
            when belongOp_in is NOTBELONG and belongOp_out is BELONG
            params::
                :param dataDictionary_in: dataframe with the data before the median
                :param dataDictionary_out: dataframe with the data after the median
                :param fixValueInput: fix value to apply the median
                :param axis_param: axis to apply the median
                :param field: field to apply the median

            Returns:
                :return: True if the median is applied correctly to the fix value
            """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the median of these numeric columns
            median_value = only_numbers_df.median().median()
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                median = dataDictionary_in[col_name].median()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                median = numeric_data.median()
                # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if value == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        # Check the dataDictionary_out positions with missing values have been replaced with the median
        median = dataDictionary_in[field].median()
        for idx, value in dataDictionary_in[field].items():
            if value == fixValueInput:
                return False

    return True


def checkFixValueMedianNotBelongNotBelong(dataDictionary_in, dataDictionary_out, fixValueInput, axis_param, field):
    """
        Check if the median is applied correctly to the fix input value when the input and output dataframes
        when belongOp_in and belongOp_out are both NOTBELONG
        params::
            :param dataDictionary_in: dataframe with the data before the median
            :param dataDictionary_out: dataframe with the data after the median
            :param fixValueInput: fix value to apply the median
            :param axis_param: axis to apply the median
            :param field: field to apply the median

        Returns:
            :return: True if the median is applied correctly to the fix value
    """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the median of these numeric columns
            median_value = only_numbers_df.median().median()
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                median = dataDictionary_in[col_name].median()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if dataDictionary_in.at[idx, col_name] == fixValueInput:
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                median = numeric_data.median()
                # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if value == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        # Check the dataDictionary_out positions with missing values have been replaced with the median
        median = dataDictionary_in[field].median()
        for idx, value in dataDictionary_in[field].items():
            if value == fixValueInput:
                return False

    return True


def checkFixValueMedian(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                        belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                        axis_param: int = None, field: str = None) -> bool:
    """
        Check if the median is applied correctly to the fix value in the dataDictionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: dataframe with the data before the median
            :param dataDictionary_out: dataframe with the data after the median
            :param fixValueInput: fix value to apply the median
            :param belongOp_in: if condition to check the invariant
            :param belongOp_out: then condition to check the invariant
            :param axis_param: axis to apply the median
            :param field: field to apply the median

        Returns:
            :return: True if median is applied correctly to the fix value
    """
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (
            belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkFixValueMedianBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                              fixValueInput=fixValueInput, belongOp_out=belongOp_out,
                                              axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkFixValueMedianNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        fixValueInput=fixValueInput, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValueMedianNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        fixValueInput=fixValueInput, axis_param=axis_param, field=field)

    return True if result else False


def checkFixValueClosestBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                                 belongOp_out: Belong, axis_param: int = None, field: str = None) -> bool:
    """
        Check if the closest is applied correctly to the fix input value
        when the input and output dataframes when belongOp_in is Belong
        params:
            :param dataDictionary_in: dataframe with the data before the closest
            :param dataDictionary_out: dataframe with the data after the closest
            :param fixValueInput: fix value to apply the closest
            :param belongOp_out: then condition to check the invariant
            :param axis_param: axis to apply the closest
            :param field: field to apply the closest

        Returns:
            :return: True if the closest is applied correctly to the fix input value
    """
    if field is None:
        if axis_param is None:
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Flatten the DataFrame into a single series of values
            flattened_values = only_numbers_df.values.flatten().tolist()

            # Find the closest numeric value to the fix value
            closest_value = find_closest_value(flattened_values, fixValueInput)

            # Replace the missing values with the closest numeric values
            for i in range(len(dataDictionary_in.index)):
                for j in range(len(dataDictionary_in.columns)):
                    current_value = dataDictionary_in.iiloc[i, j]
                    if current_value == fixValueInput:
                        if dataDictionary_out.iloc[i, j] != closest_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if pd.isnull(dataDictionary_in.iloc[i, j]):
                            raise ValueError(
                                "Error: it's not possible to apply the closest operation to the null values")
                        if (dataDictionary_out.loc[i, j] != dataDictionary_in.loc[i, j]) and not (
                                pd.isnull(dataDictionary_in.loc[i, j]) or pd.isnull(dataDictionary_out.loc[i, j])):
                            return False
        elif axis_param == 0:
            # Iterate over each column
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:

                # Flatten the column into a list of values
                flattened_values = dataDictionary_in[col_name].values.flatten().tolist()

                # Find the closest numeric value to the fix value in the column
                closest_value = find_closest_value(flattened_values, fixValueInput)

                # Replace the missing values with the closest numeric values in the column
                for i in range(len(dataDictionary_in.index)):
                    current_value = dataDictionary_in.at[i, col_name]
                    if current_value == fixValueInput:
                        if dataDictionary_out.at[i, col_name] != closest_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if pd.isnull(dataDictionary_in.at[i, col_name]):
                            raise ValueError(
                                "Error: it's not possible to apply the closest operation to the null values")
                        if (dataDictionary_out.loc[i, col_name] != dataDictionary_in.loc[i, col_name]) and not (
                                pd.isnull(dataDictionary_in.loc[i, col_name]) or pd.isnull(
                                dataDictionary_out.loc[i, col_name])):
                            return False
        elif axis_param == 1:
            # Iterate over each row
            for row_idx in range(len(dataDictionary_in.index)):
                # Get the numeric values in the current row
                numeric_values_in_row = dataDictionary_in.iloc[row_idx].select_dtypes(
                    include=[np.number]).values.tolist()

                # Flatten the row into a list of values
                flattened_values = [val for sublist in numeric_values_in_row for val in sublist]

                # Find the closest numeric value to the fix value in the row
                closest_value = find_closest_value(flattened_values, fixValueInput)

                # Replace the missing values with the closest numeric values in the row
                for col_name in dataDictionary_in.columns:
                    current_value = dataDictionary_in.at[row_idx, col_name]
                    if current_value == fixValueInput:
                        if dataDictionary_out.at[row_idx, col_name] != closest_value:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if pd.isnull(dataDictionary_in.at[
                                         row_idx, col_name]):
                            raise ValueError(
                                "Error: it's not possible to apply the closest operation to the null values")
                        if (dataDictionary_out.at[row_idx, col_name] != dataDictionary_in.at[
                            row_idx, col_name]) and not (
                                pd.isnull(dataDictionary_in.loc[row_idx, col_name]) or pd.isnull(
                                dataDictionary_out.loc[row_idx, col_name])):
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        # Flatten the column into a list of values
        flattened_values = dataDictionary_in[field].values.flatten().tolist()

        # Find the closest numeric value to the fix value in the column
        closest_value = find_closest_value(flattened_values, fixValueInput)

        # Replace the missing values with the closest numeric values in the column
        for i in range(len(dataDictionary_in.index)):
            current_value = dataDictionary_in.at[i, field]
            if current_value == fixValueInput:
                if dataDictionary_out.at[i, field] != closest_value:
                    if belongOp_out == Belong.BELONG:
                        return False
                    elif belongOp_out == Belong.NOTBELONG:
                        return True
            else:
                if pd.isnull(dataDictionary_in.at[i, field]):
                    raise ValueError(
                        "Error: it's not possible to apply the closest operation to the null values")
                if (dataDictionary_out.at[i, field] != dataDictionary_in.at[i, field]) and not (
                        pd.isnull(dataDictionary_in.loc[i, field]) or pd.isnull(
                        dataDictionary_out.loc[i, field])):
                    return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkFixValueClosestNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                                        axis_param: int = None, field: str = None) -> bool:
    """
        Check if the closest is applied correctly to the fix input value
        when the input and output dataframes when belongOp_in is NOTBELONG and belongOp_out is BELONG
        params:
            :param dataDictionary_in: dataframe with the data before the closest
            :param dataDictionary_out: dataframe with the data after the closest
            :param fixValueInput: fix value to apply the closest
            :param axis_param: axis to apply the closest
            :param field: field to apply the closest

        Returns:
            :return: True if the closest is applied correctly to the fix input value
    """
    if field is None:
        if axis_param is None:

            # Replace the missing values with the closest numeric values
            for i in range(len(dataDictionary_in.index)):
                for j in range(len(dataDictionary_in.columns)):
                    current_value = dataDictionary_in.iiloc[i, j]
                    if current_value == fixValueInput:
                        return False

        elif axis_param == 0:
            # Iterate over each column
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:

                # Replace the missing values with the closest numeric values in the column
                for i in range(len(dataDictionary_in.index)):
                    current_value = dataDictionary_in.at[i, col_name]
                    if current_value == fixValueInput:
                        return False

        elif axis_param == 1:
            # Iterate over each row
            for row_idx in range(len(dataDictionary_in.index)):

                # Replace the missing values with the closest numeric values in the row
                for col_name in dataDictionary_in.columns:
                    current_value = dataDictionary_in.at[row_idx, col_name]
                    if current_value == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Replace the missing values with the closest numeric values in the column
        for i in range(len(dataDictionary_in.index)):
            current_value = dataDictionary_in.at[i, field]
            if current_value == fixValueInput:
                return False

    return True


def checkFixValueClosestNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                                             axis_param: int = None, field: str = None) -> bool:
    """
        Check if the closest is applied correctly to the fix input value
        when the input and output dataframes when belongOp_in and belongOp_out are both NOTBELONG
        params:
            :param dataDictionary_in: dataframe with the data before the closest
            :param dataDictionary_out: dataframe with the data after the closest
            :param fixValueInput: fix value to apply the closest
            :param axis_param: axis to apply the closest
            :param field: field to apply the closest

        Returns:
            :return: True if the closest is applied correctly to the fix input value
    """
    if field is None:
        if axis_param is None:

            # Replace the missing values with the closest numeric values
            for i in range(len(dataDictionary_in.index)):
                for j in range(len(dataDictionary_in.columns)):
                    current_value = dataDictionary_in.iiloc[i, j]
                    if current_value == fixValueInput:
                        return False

        elif axis_param == 0:
            # Iterate over each column
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:

                # Replace the missing values with the closest numeric values in the column
                for i in range(len(dataDictionary_in.index)):
                    current_value = dataDictionary_in.at[i, col_name]
                    if current_value == fixValueInput:
                        return False

        elif axis_param == 1:
            # Iterate over each row
            for row_idx in range(len(dataDictionary_in.index)):

                # Replace the missing values with the closest numeric values in the row
                for col_name in dataDictionary_in.columns:
                    current_value = dataDictionary_in.at[row_idx, col_name]
                    if current_value == fixValueInput:
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        # Replace the missing values with the closest numeric values in the column
        for i in range(len(dataDictionary_in.index)):
            current_value = dataDictionary_in.at[i, field]
            if current_value == fixValueInput:
                return False

    return True


def checkFixValueClosest(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, fixValueInput,
                         belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                         axis_param: int = None, field: str = None) -> bool:
    """
        Check if the closest value is applied correctly to the fix value in the dataDictionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: dataframe with the data before the closest value
            :param dataDictionary_out: dataframe with the data after the closest value
            :param fixValueInput: fix value to apply the closest value
            :param belongOp_in: if condition to check the invariant
            :param belongOp_out: then condition to check the invariant
            :param axis_param: axis to apply the closest value
            :param field: field to apply the closest value

        Returns:
            :return: True if closest value is applied correctly to the fix value
    """
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (
            belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkFixValueClosestBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                            fixValueInput=fixValueInput, belongOp_out=belongOp_out,
                                            axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkFixValueClosestNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                    fixValueInput=fixValueInput, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkFixValueClosestNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                    fixValueInput=fixValueInput, axis_param=axis_param, field=field)

    return True if result else False


def checkIntervalInterpolation(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                       leftMargin: float, rightMargin: float, closureType: Closure,
                       belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None, field: str = None) -> bool:
    """
    Check if the interpolation is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the interpolation
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the interpolation
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the interpolation
        :param field: (str) field to apply the interpolation

    Returns:
        :return: True if the interpolation is applied correctly on the interval
    """
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkIntervalInterpolationBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalInterpolationNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalInterpolationNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)

    return True if result else False


def checkIntervalInterpolationBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        leftMargin: float, rightMargin: float, closureType: Closure,
                                        belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                                        field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframes when belongOp_in is BELONG and belongOp_out is NOTBELONG
    params:
        :param dataDictionary_in: dataframe with the data before the interpolation
        :param dataDictionary_out: dataframe with the data after the interpolation
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param closureType: closure of the interval
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param axis_param: axis to apply the interpolation
        :param field: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
    """
    dataDictionary_in_copy = dataDictionary_in.copy()
    if field is None:
        if axis_param == 0:
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                dataDictionary_in_copy[col_name] = (
                    dataDictionary_in[col_name].apply(lambda x: np.nan if check_interval_condition(x, leftMargin, rightMargin, closureType) else x).
                    interpolate(method='linear', limit_direction='both'))

            # Iterate over each column
            for col in dataDictionary_in.columns:
                # For each index in the column
                for idx in dataDictionary_in.index:
                    # Verify if the value is NaN in the original dataframe
                    if pd.isnull(dataDictionary_in.at[idx, col]):
                        # Replace the value with the corresponding one from dataDictionary_in
                        dataDictionary_in_copy.at[idx, col] = dataDictionary_in.at[idx, col]

            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                for idx in dataDictionary_in.index:
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        if dataDictionary_out.at[idx, col_name] != dataDictionary_in_copy.at[idx, col_name]:
                            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if (dataDictionary_out.at[idx, col_name] != dataDictionary_in.at[idx, col_name]) and not (
                                pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                dataDictionary_out.at[idx, col_name])):
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                dataDictionary_in_copy[row] = (
                    numeric_data[row].apply(lambda x: np.nan if check_interval_condition(x, leftMargin, rightMargin, closureType) else x).
                    interpolate(method='linear', limit_direction='both'))

            # Iterate over each column
            for col in dataDictionary_in.columns:
                # For each index in the column
                for idx in dataDictionary_in.index:
                    # Verify if the value is NaN in the original dataframe
                    if pd.isnull(dataDictionary_in.at[idx, col]):
                        # Replace the value with the corresponding one from dataDictionary_in
                        dataDictionary_in_copy.at[idx, col] = dataDictionary_in.at[idx, col]

            for idx in dataDictionary_in.index:
                for col_name in dataDictionary_in.columns:
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        if dataDictionary_out.at[idx, col_name] != dataDictionary_in_copy.at[idx, col_name]:
                            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if (dataDictionary_out.at[idx, col_name] != dataDictionary_in.at[idx, col_name]) and not (
                                pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                dataDictionary_out.at[idx, col_name])):
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field does not exist in the dataframe")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        dataDictionary_in_copy[field] = (
            dataDictionary_in[field].apply(
                lambda x: np.nan if check_interval_condition(x, leftMargin, rightMargin, closureType) else x).
            interpolate(method='linear', limit_direction='both'))


        # For each index in the column
        for idx in dataDictionary_in.index:
            # Verify if the value is NaN in the original dataframe
            if pd.isnull(dataDictionary_in.at[idx, field]):
                # Replace the value with the corresponding one from dataDictionary_in
                dataDictionary_in_copy.at[idx, field] = dataDictionary_in.at[idx, field]

        for idx in dataDictionary_in.index:
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                if dataDictionary_out.at[idx, field] != dataDictionary_in_copy.at[idx, field]:
                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                        return False
                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                        return True
            else:
                if (dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]) and not (
                        pd.isnull(dataDictionary_out.at[idx, field]) or pd.isnull(
                    dataDictionary_out.at[idx, field])):
                    return False

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        return True
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkIntervalInterpolationNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        leftMargin: float, rightMargin: float, closureType: Closure,
                                        belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                                        field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframes when belongOp_in is BELONG and belongOp_out is NOTBELONG
    params:
        :param dataDictionary_in: dataframe with the data before the interpolation
        :param dataDictionary_out: dataframe with the data after the interpolation
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param closureType: closure of the interval
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param axis_param: axis to apply the interpolation
        :param field: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
    """
    dataDictionary_in_copy = dataDictionary_in.copy()
    if field is None:
        if axis_param == 0:
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                for idx in dataDictionary_in.index:
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        return False
        elif axis_param == 1:
            for idx in dataDictionary_in.iterrows():
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        for idx in dataDictionary_in.index:
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                return False

    return True


def checkIntervalInterpolationNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        leftMargin: float, rightMargin: float, closureType: Closure,
                                        belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                                        field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframes when belongOp_in is BELONG and belongOp_out is NOTBELONG
    params:
        :param dataDictionary_in: dataframe with the data before the interpolation
        :param dataDictionary_out: dataframe with the data after the interpolation
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param closureType: closure of the interval
        :param belongOp_in: if condition to check the invariant
        :param belongOp_out: then condition to check the invariant
        :param axis_param: axis to apply the interpolation
        :param field: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
    """
    dataDictionary_in_copy = dataDictionary_in.copy()
    if field is None:
        if axis_param == 0:
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                for idx in dataDictionary_in.index:
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        return False
        elif axis_param == 1:
            for idx in dataDictionary_in.iterrows():
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        for idx in dataDictionary_in.index:
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                return False

    return True


def checkIntervalMean(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                       leftMargin: float, rightMargin: float, closureType: Closure,
                       belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None, field: str = None) -> bool:
    """
    Check if the mean is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the mean
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the mean
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the mean
        :param field: (str) field to apply the mean

    Returns:
        :return: True if the mean is applied correctly on the interval
    """
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkIntervalMeanBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalMeanNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalMeanNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)

    return True if result else False


def checkIntervalMeanBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the mean is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the mean
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the mean
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the mean
        :param field: (str) field to apply the mean

    Returns:
        :return: True if the mean is applied correctly on the interval
    """

    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the mean of these numeric columns
            mean_value = only_numbers_df.mean().mean()
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                            if dataDictionary_out.at[idx, col_name] != mean_value:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                    dataDictionary_out.at[idx, col_name])):
                                return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                mean = dataDictionary_in[col_name].mean()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                            if dataDictionary_out.at[idx, col_name] != mean:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                    dataDictionary_out.at[idx, col_name])):
                                return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                mean = numeric_data.mean()
                # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        if dataDictionary_out.at[idx, col_name] != mean:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                dataDictionary_out.at[idx, col_name])):
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the mean
        mean = dataDictionary_in[field].mean()
        for idx, value in dataDictionary_in[field].items():
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                if dataDictionary_out.at[idx, field] != mean:
                    if belongOp_out == Belong.BELONG:
                        return False
                    elif belongOp_out == Belong.NOTBELONG:
                        return True
            else:
                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                    return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkIntervalMeanNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the mean is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the mean
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the mean
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the mean
        :param field: (str) field to apply the mean

    Returns:
        :return: True if the mean is applied correctly on the interval
    """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the mean of these numeric columns
            mean_value = only_numbers_df.mean().mean()
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                            return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                mean = dataDictionary_in[col_name].mean()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                mean = numeric_data.mean()
                # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the mean
        mean = dataDictionary_in[field].mean()
        for idx, value in dataDictionary_in[field].items():
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                return False

        return True


def checkIntervalMeanNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the mean is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the mean
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the mean
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the mean
        :param field: (str) field to apply the mean

    Returns:
        :return: True if the mean is applied correctly on the interval
    """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the mean of these numeric columns
            mean_value = only_numbers_df.mean().mean()
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin,
                                                    closureType):
                            return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                mean = dataDictionary_in[col_name].mean()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin,
                                                    closureType):
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                mean = numeric_data.mean()
                # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin,
                                                closureType):
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the mean
        mean = dataDictionary_in[field].mean()
        for idx, value in dataDictionary_in[field].items():
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                return False

        return True


def checkIntervalMedian(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                       leftMargin: float, rightMargin: float, closureType: Closure,
                       belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None, field: str = None) -> bool:
    """
    Check if the median is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the median
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the median
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the median
        :param field: (str) field to apply the median

    Returns:
        :return: True if the median is applied correctly on the interval
    """
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkIntervalMedianBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalMedianNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalMedianNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)

    return True if result else False


def checkIntervalMedianBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the median is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the median
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the median
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the median
        :param field: (str) field to apply the median

    Returns:
        :return: True if the median is applied correctly on the interval
    """

    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the median of these numeric columns
            median_value = only_numbers_df.median().median()
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                            if dataDictionary_out.at[idx, col_name] != median_value:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                    dataDictionary_out.at[idx, col_name])):
                                return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                median = dataDictionary_in[col_name].median()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                            if dataDictionary_out.at[idx, col_name] != median:
                                if belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                    dataDictionary_out.at[idx, col_name])):
                                return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                median = numeric_data.median()
                # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        if dataDictionary_out.at[idx, col_name] != median:
                            if belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(
                                dataDictionary_out.at[idx, col_name])):
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the median
        median = dataDictionary_in[field].median()
        for idx, value in dataDictionary_in[field].items():
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                if dataDictionary_out.at[idx, field] != median:
                    if belongOp_out == Belong.BELONG:
                        return False
                    elif belongOp_out == Belong.NOTBELONG:
                        return True
            else:
                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                    return False

    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkIntervalMedianNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the median is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the median
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the median
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the median
        :param field: (str) field to apply the median

    Returns:
        :return: True if the median is applied correctly on the interval
    """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the median of these numeric columns
            median_value = only_numbers_df.median().median()
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                            return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                median = dataDictionary_in[col_name].median()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                median = numeric_data.median()
                # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin, closureType):
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the median
        median = dataDictionary_in[field].median()
        for idx, value in dataDictionary_in[field].items():
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                return False

        return True


def checkIntervalMedianNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the median is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the median
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the median
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the median
        :param field: (str) field to apply the median

    Returns:
        :return: True if the median is applied correctly on the interval
    """
    if field is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
            # Calculate the median of these numeric columns
            median_value = only_numbers_df.median().median()
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.columns:
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin,
                                                    closureType):
                            return False
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                median = dataDictionary_in[col_name].median()
                for idx, value in dataDictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin,
                                                    closureType):
                            return False
        elif axis_param == 1:
            for idx, row in dataDictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                median = numeric_data.median()
                # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                for col_name, value in numeric_data.items():
                    if check_interval_condition(dataDictionary_in.at[idx, col_name], leftMargin, rightMargin,
                                                closureType):
                        return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the dataDictionary_out positions with missing values have been replaced with the median
        median = dataDictionary_in[field].median()
        for idx, value in dataDictionary_in[field].items():
            if check_interval_condition(dataDictionary_in.at[idx, field], leftMargin, rightMargin, closureType):
                return False

        return True


def checkIntervalClosest(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                       leftMargin: float, rightMargin: float, closureType: Closure,
                       belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None, field: str = None) -> bool:
    """
    Check if the closest is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the closest
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the closest
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the closest
        :param field: (str) field to apply the closest

    Returns:
        :return: True if the closest is applied correctly on the interval
    """
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkIntervalClosestBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkIntervalClosestNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkIntervalClosestNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                  leftMargin=leftMargin, rightMargin=rightMargin,
                                                  closureType=closureType, belongOp_in=belongOp_in,
                                                  belongOp_out=belongOp_out, axis_param=axis_param, field=field)

    return True if result else False


def checkIntervalClosestBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the closest is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the closest
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the closest
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the closest
        :param field: (str) field to apply the closest

    Returns:
        :return: True if the closest is applied correctly on the interval
    """
    #TODO: Implement this function
    if belongOp_out == Belong.BELONG:
        return True
    elif belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkIntervalClosestNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the closest is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the closest
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the closest
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the closest
        :param field: (str) field to apply the closest

    Returns:
        :return: True if the closest is applied correctly on the interval
    """
    #TODO: Implement this function
    return True


def checkIntervalClosestNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            leftMargin: float, rightMargin: float, closureType: Closure,
                            belongOp_in: Belong, belongOp_out: Belong, axis_param: int = None,
                            field: str = None) -> bool:
    """
    Check if the closest is applied correctly on the interval
    to the dataDictionary_out respect to the dataDictionary_in

    params:
        :param dataDictionary_in: (pd.DataFrame) dataframe with the data before the closest
        :param dataDictionary_out: (pd.DataFrame) dataframe with the data after the closest
        :param leftMargin: (float) left margin of the interval
        :param rightMargin: (float) right margin of the interval
        :param closureType: (Closure) closure of the interval
        :param belongOp_in: (Belong) if condition to check the invariant
        :param belongOp_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the closest
        :param field: (str) field to apply the closest

    Returns:
        :return: True if the closest is applied correctly on the interval
    """
    #TODO: Implement this function
    return True


def checkSpecialTypeInterpolation(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                  specialTypeInput: SpecialType,
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

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkSpecialTypeInterpolationBelong(dataDictionary_in=dataDictionary_in,
                                                     dataDictionary_out=dataDictionary_out,
                                                     specialTypeInput=specialTypeInput,
                                                     belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                                     dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                     missing_values=missing_values, axis_param=axis_param,
                                                     field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkSpecialTypeInterpolationNotBelongBelong(dataDictionary_in=dataDictionary_in,
                                                              dataDictionary_out=dataDictionary_out,
                                                              specialTypeInput=specialTypeInput,
                                                              dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                              missing_values=missing_values, axis_param=axis_param,
                                                              field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkSpecialTypeInterpolationNotBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                                 dataDictionary_out=dataDictionary_out,
                                                                 specialTypeInput=specialTypeInput,
                                                                 dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                                 missing_values=missing_values, axis_param=axis_param,
                                                                 field=field)

    return True if result else False


def checkSpecialTypeInterpolationBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        specialTypeInput: SpecialType,
                                        belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                        dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                        axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframe
    when belongOp_in is BELONG and belongOp_out is BELONG or NOTBELONG
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
                    dataDictionary_in_copy[col_name] = (
                        dataDictionary_in[col_name].apply(lambda x: np.nan if x in missing_values else x).
                        interpolate(method='linear', limit_direction='both'))

                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx in dataDictionary_in.index:
                        if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                            if dataDictionary_out.at[idx, col_name] != dataDictionary_in_copy.at[idx, col_name]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.at[idx, col_name] != dataDictionary_in.at[idx, col_name]) and not(pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(dataDictionary_out.at[idx, col_name])):
                                return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    dataDictionary_in_copy[row] = (
                        numeric_data[row].apply(lambda x: np.nan if x in missing_values else x).
                        interpolate(method='linear', limit_direction='both'))
                for col_name in dataDictionary_in.columns:
                    for idx in dataDictionary_in.index:
                        if dataDictionary_in.at[idx, col_name] in missing_values or pd.isnull(dataDictionary_in.at[idx, col_name]):
                            if dataDictionary_out.at[idx, col_name] != dataDictionary_in_copy.at[idx, col_name]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.at[idx, col_name] != dataDictionary_in.at[idx, col_name]) and not(pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(dataDictionary_out.at[idx, col_name])):
                                return False

        elif specialTypeInput == SpecialType.INVALID:
            if axis_param == 0:
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    dataDictionary_in_copy[col] = (
                        dataDictionary_in[col].apply(lambda x: np.nan if x in missing_values else x).
                        interpolate(method='linear', limit_direction='both'))

                # Iterate over each column
                for col in dataDictionary_in.columns:
                    # For each index in the column
                    for idx in dataDictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(dataDictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            dataDictionary_in_copy.at[idx, col] = dataDictionary_in.at[idx, col]

                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx in dataDictionary_in.index:
                        if dataDictionary_in.at[idx, col] in missing_values:
                            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                return True
                        else:
                            if (dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]) and not(pd.isnull(dataDictionary_out.at[idx, col]) or pd.isnull(dataDictionary_out.at[idx, col])):
                                return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    dataDictionary_in_copy[row] = (
                        numeric_data[row].apply(lambda x: np.nan if x in missing_values else x).
                        interpolate(method='linear', limit_direction='both'))

                # Iterate over each column
                for col in dataDictionary_in.columns:
                    # For each index in the column
                    for idx in dataDictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(dataDictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            dataDictionary_in_copy.at[idx, col] = dataDictionary_in.at[idx, col]

                for col in dataDictionary_in.columns:
                    for idx in dataDictionary_in.index:
                        if dataDictionary_in.at[idx, col] in missing_values:
                            if dataDictionary_out.at[idx, col] != dataDictionary_in_copy.at[idx, col]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]) and not(pd.isnull(dataDictionary_out.at[idx, col]) or pd.isnull(dataDictionary_out.at[idx, col])):
                                return False

        elif specialTypeInput == SpecialType.OUTLIER:
            if axis_param == 0:
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in dataDictionary_in[col].items():
                        if dataDictionary_outliers_mask.at[idx, col] == 1:
                            dataDictionary_in_copy.at[idx, col] = np.NaN
                    dataDictionary_in_copy[col] = dataDictionary_in_copy[col].interpolate(method='linear', limit_direction='both')

                # Iterate over each column
                for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    # For each index in the column
                    for idx in dataDictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(dataDictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            dataDictionary_in_copy.at[idx, col] = dataDictionary_in.at[idx, col]

                for col in dataDictionary_outliers_mask.columns:
                    for idx in dataDictionary_outliers_mask.index:
                        if dataDictionary_outliers_mask.at[idx, col] == 1:
                            if dataDictionary_out.at[idx, col] != dataDictionary_in_copy.at[idx, col]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]) and not(pd.isnull(dataDictionary_out.at[idx, col]) or pd.isnull(dataDictionary_out.at[idx, col])):
                                return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                        if dataDictionary_outliers_mask.at[idx, col] == 1:
                            dataDictionary_in_copy.at[idx, col] = np.NaN
                    # Interpolate the row
                    dataDictionary_in_copy.loc[idx] = dataDictionary_in_copy.loc[idx].interpolate(method='linear', limit_direction='both')

                # Iterate over each column
                for col in dataDictionary_in.columns:
                    # For each index in the column
                    for idx in dataDictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(dataDictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            dataDictionary_in_copy.at[idx, col] = dataDictionary_in.at[idx, col]

                for col in dataDictionary_outliers_mask.columns:
                    for idx in dataDictionary_outliers_mask.index:
                        if dataDictionary_outliers_mask.at[idx, col] == 1:
                            if dataDictionary_out.at[idx, col] != dataDictionary_in_copy.at[idx, col]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col]) and not(pd.isnull(dataDictionary_out.at[idx, col]) or pd.isnull(dataDictionary_out.at[idx, col])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        if specialTypeInput == SpecialType.MISSING:
            dataDictionary_in_copy[field] = (dataDictionary_in[field].apply(lambda x: np.nan if x in missing_values else x).
                                             interpolate(method='linear', limit_direction='both'))

            for idx in dataDictionary_in.index:
                if dataDictionary_in.at[idx, field] in missing_values or pd.isnull(
                        dataDictionary_in.at[idx, field]):
                    if dataDictionary_out.at[idx, field] != dataDictionary_in_copy.at[idx, field]:
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return True
                else:
                    if (dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]) and not(pd.isnull(dataDictionary_out.at[idx, field]) or pd.isnull(dataDictionary_out.at[idx, field])):
                        return False

        elif specialTypeInput == SpecialType.INVALID:
            dataDictionary_in_copy[field] = (dataDictionary_in[field].apply(lambda x: np.nan if x in missing_values
                else x).interpolate(method='linear', limit_direction='both'))

            # For each index in the column
            for idx in dataDictionary_in.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(dataDictionary_in.at[idx, field]):
                    # Replace the value with the corresponding one from dataDictionary_copy_copy
                    dataDictionary_in_copy.at[idx, field] = dataDictionary_in.at[idx, field]

            for idx in dataDictionary_in_copy.index:
                if dataDictionary_in.at[idx, field] in missing_values:
                    if dataDictionary_out.at[idx, field] != dataDictionary_in_copy.at[idx, field]:
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return True
                else:
                    if (dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]) and not(pd.isnull(dataDictionary_out.at[idx, field]) or pd.isnull(dataDictionary_out.at[idx, field])):
                        return False

        elif specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_in[field].items():
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    dataDictionary_in_copy.at[idx, field] = np.NaN
            dataDictionary_in_copy[field] = dataDictionary_in_copy[field].interpolate(method='linear', limit_direction='both')

            # For each index in the column
            for idx in dataDictionary_in.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(dataDictionary_in.at[idx, field]):
                    # Replace the value with the corresponding one from dataDictionary_copy_copy
                    dataDictionary_in_copy.at[idx, field] = dataDictionary_in.at[idx, field]

            for idx in dataDictionary_outliers_mask.index:
                if dataDictionary_outliers_mask.at[idx, field] == 1:
                    if dataDictionary_out.at[idx, field] != dataDictionary_in_copy.at[idx, field]:
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return True
                else:
                    if (dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field]) and not(pd.isnull(dataDictionary_out.at[idx, field]) or pd.isnull(dataDictionary_out.at[idx, field])):
                        return False

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        return True
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkSpecialTypeInterpolationNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
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
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
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


def checkSpecialTypeInterpolationNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
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
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
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


def checkSpecialTypeMean(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                         specialTypeInput: SpecialType,
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

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkSpecialTypeMeanBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                            specialTypeInput=specialTypeInput,
                                            belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                            dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                            missing_values=missing_values, axis_param=axis_param,
                                            field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkSpecialTypeMeanNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                     specialTypeInput=specialTypeInput,
                                                     dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                     missing_values=missing_values, axis_param=axis_param,
                                                     field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkSpecialTypeMeanNotBelongNotBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        specialTypeInput=specialTypeInput,
                                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                        missing_values=missing_values, axis_param=axis_param,
                                                        field=field)

    return True if result else False


def checkSpecialTypeMeanBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                               specialTypeInput: SpecialType,
                               belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
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
                                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                        return True
                            else:
                                if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]) and not(pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(dataDictionary_out.at[idx, col_name])):
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
                                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                        return True
                            else:
                                if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]) and not(pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(dataDictionary_out.at[idx, col_name])):
                                    return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    mean = numeric_data.mean()
                    # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.at[idx, col_name] != mean:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]) and not(pd.isnull(dataDictionary_out.at[idx, col_name]) or pd.isnull(dataDictionary_out.at[idx, col_name])):
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
                                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                        return True
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
                                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                        return True
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                    return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    mean = numeric_data.mean()
                    # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values:
                            if dataDictionary_out.at[idx, col_name] != mean:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
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
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
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
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                dataDictionary_out.loc[idx, col])):
                                return False
            elif axis_param == 1:  # Iterate over each row
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    mean = numeric_data.mean()
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            if dataDictionary_out.at[idx, col_name] != mean:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and
                                    pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if specialTypeInput == SpecialType.MISSING:
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            mean = dataDictionary_in[field].mean()
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values or pd.isnull(value):
                    if dataDictionary_out.at[idx, field] != mean:
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False
        if specialTypeInput == SpecialType.INVALID:
            # Check the dataDictionary_out positions with missing values have been replaced with the mean
            mean = dataDictionary_in[field].mean()
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values:
                    if dataDictionary_out.at[idx, field] != mean:
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return True
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
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return  True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                        dataDictionary_out.loc[idx, field])):
                        return False

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        return True
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkSpecialTypeMeanNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        specialTypeInput: SpecialType,
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
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


def checkSpecialTypeMeanNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                           specialTypeInput: SpecialType,
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    # Check if the missing values in the row have been replaced with the mean in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
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


def checkSpecialTypeMedian(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                           specialTypeInput: SpecialType,
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

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkSpecialTypeMedianBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                              specialTypeInput=specialTypeInput,
                                              belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                              dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                              missing_values=missing_values, axis_param=axis_param,
                                              field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkSpecialTypeMedianNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                       specialTypeInput=specialTypeInput,
                                                       dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                       missing_values=missing_values, axis_param=axis_param,
                                                       field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkSpecialTypeMedianNotBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                          dataDictionary_out=dataDictionary_out,
                                                          specialTypeInput=specialTypeInput,
                                                          dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                          missing_values=missing_values, axis_param=axis_param,
                                                          field=field)

    return True if result else False


def checkSpecialTypeMedianBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                 specialTypeInput: SpecialType,
                                 belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
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
                                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                        return True
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
                                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                        return True
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name]:
                                    return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    median = numeric_data.median()
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            if dataDictionary_out.at[idx, col_name] != median:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
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
                                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                        return True
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
                                    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                        return False
                                    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                        return True
                            else:
                                if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[
                                    idx, col_name] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col_name]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col_name])):
                                    return False
            elif axis_param == 1:
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    median = numeric_data.median()
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values:
                            if dataDictionary_out.at[idx, col_name] != median:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
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
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
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
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, col] != dataDictionary_in.loc[idx, col] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                dataDictionary_out.loc[idx, col])):
                                return False
            elif axis_param == 1:  # Iterate over each row
                for idx, row in dataDictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    median = numeric_data.median()
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            if dataDictionary_out.at[idx, col_name] != median:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, col_name] != dataDictionary_in.loc[idx, col_name] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col_name]) and
                                    pd.isnull(dataDictionary_out.loc[idx, col_name])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if specialTypeInput == SpecialType.MISSING:
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            median = dataDictionary_in[field].median()
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values or pd.isnull(value):
                    if dataDictionary_out.at[idx, field] != median:
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                        return False
        if specialTypeInput == SpecialType.INVALID:
            # Check the dataDictionary_out positions with missing values have been replaced with the median
            median = dataDictionary_in[field].median()
            for idx, value in dataDictionary_in[field].items():
                if value in missing_values:
                    if dataDictionary_out.at[idx, field] != median:
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return True
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
                        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                            return False
                        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                            return True
                else:
                    if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                            pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(dataDictionary_out.loc[idx, field])):
                        return False

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        return True
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkSpecialTypeMedianNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                          specialTypeInput: SpecialType,
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
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


def checkSpecialTypeMedianNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                             specialTypeInput: SpecialType,
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    # Check if the missing values in the row have been replaced with the median in dataDictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            return False
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
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
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    for col_name, value in numeric_data.items():
                        if dataDictionary_outliers_mask.at[idx, col_name] == 1:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
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


def checkSpecialTypeClosest(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                            specialTypeInput: SpecialType,
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
    result = True

    if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
        result = checkSpecialTypeClosestBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                               specialTypeInput=specialTypeInput, belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                               dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                               missing_values=missing_values, axis_param=axis_param,
                                               field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkSpecialTypeClosestNotBelongBelong(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        specialTypeInput=specialTypeInput, belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                        missing_values=missing_values, axis_param=axis_param,
                                                        field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkSpecialTypeClosestNotBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                           dataDictionary_out=dataDictionary_out,
                                                           specialTypeInput=specialTypeInput, belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                                           dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                           missing_values=missing_values, axis_param=axis_param,
                                                           field=field)

    return True if result else False


def checkSpecialTypeClosestBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                  specialTypeInput: SpecialType, belongOp_in: Belong, belongOp_out: Belong,
                                  dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                  axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type closest is applied correctly when the input and output dataframes when belongOp_in is Belong
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
    if field is None:
        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                only_numbers_df=dataDictionary_in.select_dtypes(include=[np.number])
                # Flatten the DataFrame into a single series of values
                flattened_values = only_numbers_df.values.flatten().tolist()

                # Create a dictionary to store the closest value for each missing value
                closest_values = {}

                # For each missing value, find the closest numeric value in the flattened series
                for missing_value in missing_values:
                    if missing_value not in closest_values:
                        closest_values[missing_value] = find_closest_value(flattened_values, missing_value)

                # Replace the missing values with the closest numeric values
                for i in range(len(dataDictionary_in.index)):
                    for j in range(len(dataDictionary_in.columns)):
                        current_value = dataDictionary_in.iiloc[i, j]
                        if current_value in closest_values:
                            if dataDictionary_out.iloc[i, j] != closest_values[current_value]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if pd.isnull(dataDictionary_in.iloc[i, j]) and specialTypeInput == SpecialType.MISSING:
                                raise ValueError("Error: it's not possible to apply the closest operation to the null values")
                            if (dataDictionary_out.loc[i, j] != dataDictionary_in.loc[i, j])  and not(pd.isnull(dataDictionary_in.loc[i, j]) or pd.isnull(dataDictionary_out.loc[i, j])):
                                return False
            elif axis_param == 0:
                # Iterate over each column
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    # Get the missing values in the current column
                    missing_values_in_col = [val for val in missing_values if val in dataDictionary_in[col_name].values]

                    # If there are no missing values in the column, skip the rest of the loop
                    if not missing_values_in_col:
                        continue

                    # Flatten the column into a list of values
                    flattened_values = dataDictionary_in[col_name].values.flatten().tolist()

                    # Create a dictionary to store the closest value for each missing value
                    closest_values = {}

                    # For each missing value IN the column (more efficient), find the closest numeric value in the flattened series
                    for missing_value in missing_values_in_col:
                        if missing_value not in closest_values:
                            closest_values[missing_value] = find_closest_value(flattened_values, missing_value)

                    # Replace the missing values with the closest numeric values in the column
                    for i in range(len(dataDictionary_in.index)):
                        current_value = dataDictionary_in.at[i, col_name]
                        if current_value in closest_values:
                            if dataDictionary_out.at[i, col_name] != closest_values[current_value]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if pd.isnull(dataDictionary_in.at[i, col_name]) and specialTypeInput == SpecialType.MISSING:
                                raise ValueError("Error: it's not possible to apply the closest operation to the null values")
                            if (dataDictionary_out.loc[i, col_name] != dataDictionary_in.loc[i, col_name]) and not(pd.isnull(dataDictionary_in.loc[i, col_name]) or pd.isnull(dataDictionary_out.loc[i, col_name])):
                                return False
            elif axis_param == 1:
                # Iterate over each row
                for row_idx in range(len(dataDictionary_in.index)):
                    # Get the numeric values in the current row
                    numeric_values_in_row = dataDictionary_in.iloc[row_idx].select_dtypes(include=[np.number]).values.tolist()

                    # Get the missing values in the current row
                    missing_values_in_row = [val for val in missing_values if val in numeric_values_in_row]

                    # If there are no missing values in the row, skip the rest of the loop
                    if not missing_values_in_row and not pd.isnull(dataDictionary_in.iloc[row_idx]).any():
                        continue

                    # Create a dictionary to store the closest value for each missing value
                    closest_values = {}

                    # For each missing value IN the row (more efficient), find the closest numeric value in the numeric values
                    for missing_value in missing_values_in_row:
                        if missing_value not in closest_values:
                            closest_values[missing_value] = find_closest_value(numeric_values_in_row, missing_value)

                    # Replace the missing values with the closest numeric values in the row
                    for col_name in dataDictionary_in.columns:
                        current_value = dataDictionary_in.at[row_idx, col_name]
                        if current_value in closest_values:
                            if dataDictionary_out.at[row_idx, col_name] != closest_values[current_value]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if pd.isnull(dataDictionary_in.at[row_idx, col_name]) and specialTypeInput == SpecialType.MISSING:
                                raise ValueError("Error: it's not possible to apply the closest operation to the null values")
                            if (dataDictionary_out.at[row_idx, col_name] != dataDictionary_in.at[row_idx, col_name]) and not(pd.isnull(dataDictionary_in.loc[row_idx, col_name]) or pd.isnull(dataDictionary_out.loc[row_idx, col_name])):
                                return False

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                only_numbers_df = dataDictionary_in.select_dtypes(include=[np.number])
                # Flatten the DataFrame into a single series of values
                flattened_values = only_numbers_df.values.flatten().tolist()

                # Create a dictionary to store the closest value for each outlier value
                closest_values = {}

                # For each outlier value, find the closest numeric value in the flattened series
                for i in range(len(dataDictionary_in.index)):
                    for j in range(len(dataDictionary_in.columns)):
                        if dataDictionary_outliers_mask.iloc[i, j] == 1:
                            current_value = dataDictionary_in.iiloc[i, j]
                            if current_value not in closest_values:
                                closest_values[current_value] = find_closest_value(flattened_values, current_value)

                # Replace the outlier values with the closest numeric values
                for i in range(len(dataDictionary_in.index)):
                    for j in range(len(dataDictionary_in.columns)):
                        if dataDictionary_outliers_mask.iloc[i, j] == 1:
                            current_value = dataDictionary_in.iiloc[i, j]
                            if dataDictionary_out.iloc[i, j] != closest_values[current_value]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[i, j] != dataDictionary_in.loc[i, j]) and not(pd.isnull(dataDictionary_in.loc[i, j]) or pd.isnull(dataDictionary_out.loc[i, j])):
                                return False
            elif axis_param == 0:
                # Iterate over each column
                for col_name in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                    # Get the outlier values in the current column
                    outlier_values_in_col = [dataDictionary_in.at[i, col_name] for i in range(len(dataDictionary_in.index))
                                             if dataDictionary_outliers_mask.at[i, col_name] == 1]

                    # If there are no outlier values in the column, skip the rest of the loop
                    if not outlier_values_in_col:
                        continue

                    # Flatten the column into a list of values
                    flattened_values = dataDictionary_in[col_name].values.flatten().tolist()

                    # Create a dictionary to store the closest value for each outlier value
                    closest_values = {}

                    # For each outlier value IN the column (more efficient), find the closest numeric value in the flattened series
                    for outlier_value in outlier_values_in_col:
                        if outlier_value not in closest_values:
                            closest_values[outlier_value] = find_closest_value(flattened_values, outlier_value)

                    # Replace the outlier values with the closest numeric values in the column
                    for i in range(len(dataDictionary_in.index)):
                        current_value = dataDictionary_in.at[i, col_name]
                        if dataDictionary_outliers_mask.at[i, col_name] == 1:
                            if dataDictionary_out.at[i, col_name] != closest_values[current_value]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.loc[i, col_name] != dataDictionary_in.loc[i, col_name]) and not(pd.isnull(dataDictionary_in.loc[i, col_name]) or pd.isnull(dataDictionary_out.loc[i, col_name])):
                                return False
            elif axis_param == 1:
                # Iterate over each row
                for row_idx in range(len(dataDictionary_in.index)):
                    # Get the numeric values in the current row
                    numeric_values_in_row = dataDictionary_in.iloc[row_idx].select_dtypes(include=[np.number]).values.tolist()

                    # Get the outlier values in the current row
                    outlier_values_in_row = [dataDictionary_in.at[row_idx, col_name] for col_name in dataDictionary_in.columns
                                             if dataDictionary_outliers_mask.at[row_idx, col_name] == 1]

                    # If there are no outlier values in the row, skip the rest of the loop
                    if not outlier_values_in_row:
                        continue

                    # Create a dictionary to store the closest value for each outlier value
                    closest_values = {}

                    # For each outlier value IN the row (more efficient), find the closest numeric value in the numeric values
                    for outlier_value in outlier_values_in_row:
                        if outlier_value not in closest_values:
                            closest_values[outlier_value] = find_closest_value(numeric_values_in_row, outlier_value)

                    # Replace the outlier values with the closest numeric values in the row
                    for col_name in dataDictionary_in.columns:
                        current_value = dataDictionary_in.at[row_idx, col_name]
                        if dataDictionary_outliers_mask.at[row_idx, col_name] == 1:
                            if dataDictionary_out.at[row_idx, col_name] != closest_values[current_value]:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else:
                            if (dataDictionary_out.at[row_idx, col_name] != dataDictionary_in.at[row_idx, col_name]) and not(pd.isnull(dataDictionary_in.loc[i, col_name]) or pd.isnull(dataDictionary_out.loc[i, col_name])):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            # Get the missing values in the current column
            missing_values_in_col = [val for val in missing_values if val in dataDictionary_in[field].values]

            # If there are no missing values in the column, skip the rest of the loop
            if missing_values_in_col or pd.isnull(dataDictionary_in[field]).any():
                # Flatten the column into a list of values
                flattened_values = dataDictionary_in[field].values.flatten().tolist()

                # Create a dictionary to store the closest value for each missing value
                closest_values = {}

                # For each missing value IN the column (more efficient), find the closest numeric value in the flattened series
                for missing_value in missing_values_in_col:
                    if missing_value not in closest_values:
                        closest_values[missing_value] = find_closest_value(flattened_values, missing_value)

                # Replace the missing values with the closest numeric values in the column
                for i in range(len(dataDictionary_in.index)):
                    current_value = dataDictionary_in.at[i, field]
                    if current_value in closest_values:
                        if dataDictionary_out.at[i, field] != closest_values[current_value]:
                            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if pd.isnull(dataDictionary_in.at[i, field]) and specialTypeInput == SpecialType.MISSING:
                                raise ValueError("Error: it's not possible to apply the closest operation to the null values")
                        if (dataDictionary_out.at[i, field] != dataDictionary_in.at[i, field]) and not(pd.isnull(dataDictionary_in.loc[i, field]) or pd.isnull(dataDictionary_out.loc[i, field])):
                            return False

        if specialTypeInput == SpecialType.OUTLIER:
            # Get the outlier values in the current column
            outlier_values_in_col = [dataDictionary_in.at[i, field] for i in range(len(dataDictionary_in.index))
                                     if dataDictionary_outliers_mask.at[i, field] == 1]

            # If there are no outlier values in the column, skip the rest of the loop
            if outlier_values_in_col:
                # Flatten the column into a list of values
                flattened_values = dataDictionary_in[field].values.flatten().tolist()

                # Create a dictionary to store the closest value for each outlier value
                closest_values = {}

                # For each outlier value IN the column (more efficient), find the closest numeric value in the flattened series
                for outlier_value in outlier_values_in_col:
                    if outlier_value not in closest_values:
                        closest_values[outlier_value] = find_closest_value(flattened_values, outlier_value)

                # Replace the outlier values with the closest numeric values in the column
                for i in range(len(dataDictionary_in.index)):
                    current_value = dataDictionary_in.at[i, field]
                    if dataDictionary_outliers_mask.at[i, field] == 1:
                        if dataDictionary_out.at[i, field] != closest_values[current_value]:
                            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                return True
                    else:
                        if (dataDictionary_out.at[i, field] != dataDictionary_in.at[i, field]) and not(pd.isnull(dataDictionary_in.loc[i, field]) or pd.isnull(dataDictionary_out.loc[i, field])):
                            return False

    if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
        return True
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        return False
    else:
        return True


def checkSpecialTypeClosestNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                           specialTypeInput: SpecialType, belongOp_in: Belong, belongOp_out: Belong,
                                           dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                           axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type closest is applied correctly when the input and output dataframes when belongOp_in is NotBelong and belongOp_out is Belong
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
    if field is None:
        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            if axis_param is None or axis_param == 0 or axis_param == 1:
                # Replace the missing values with the closest numeric values
                for i in range(len(dataDictionary_in.index)):
                    for j in range(len(dataDictionary_in.columns)):
                        for missing_value in missing_values:
                            if dataDictionary_in.iloc[i, j] == missing_value or (pd.isnull(dataDictionary_in.iloc[i, j]) and specialTypeInput == SpecialType.MISSING):
                                return False

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None or axis_param == 0 or axis_param == 1:
                for i in range(len(dataDictionary_in.index)):
                    for j in range(len(dataDictionary_in.columns)):
                        if dataDictionary_outliers_mask.iloc[i, j] == 1:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            for i in range(len(dataDictionary_in.index)):
                for missing_value in missing_values:
                    if dataDictionary_in.at[i, field] == missing_value or (pd.isnull(dataDictionary_in.at[i, field]) and specialTypeInput == SpecialType.MISSING):
                        return False

        if specialTypeInput == SpecialType.OUTLIER:
            for i in range(len(dataDictionary_in.index)):
                if dataDictionary_outliers_mask.at[i, field] == 1:
                    return False

    return True


def checkSpecialTypeClosestNotBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                              specialTypeInput: SpecialType, belongOp_in: Belong, belongOp_out: Belong,
                                              dataDictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                              axis_param: int = None, field: str = None) -> bool:
    """
    Check if the special type closest is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are NotBelong
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
    if field is None:
        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            if axis_param is None or axis_param == 0 or axis_param == 1:
                for i in range(len(dataDictionary_in.index)):
                    for j in range(len(dataDictionary_in.columns)):
                        for missing_value in missing_values:
                            if dataDictionary_in.iloc[i, j] == missing_value or (pd.isnull(dataDictionary_in.iloc[i, j]) and specialTypeInput == SpecialType.MISSING):
                                return False

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None or axis_param == 0 or axis_param == 1:
                for i in range(len(dataDictionary_in.index)):
                    for j in range(len(dataDictionary_in.columns)):
                        if dataDictionary_outliers_mask.iloc[i, j] == 1:
                            return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            for i in range(len(dataDictionary_in.index)):
                for missing_value in missing_values:
                    if dataDictionary_in.at[i, field] == missing_value or (pd.isnull(dataDictionary_in.at[i, field]) and specialTypeInput == SpecialType.MISSING):
                        return False

        if specialTypeInput == SpecialType.OUTLIER:
            for i in range(len(dataDictionary_in.index)):
                if dataDictionary_outliers_mask.at[i, field] == 1:
                    return False

    return True


def checkMostFrequentBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                  specialTypeInput: SpecialType, missing_values: list, axis_param: int,
                                  field: str) -> bool:
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
                        if pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_name] != most_frequent:
                                return False
                        elif missing_values is None:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False
                        else:
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, column_name] != most_frequent:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False

            else:  # It works for invalid values and outliers
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
                                    dataDictionary_out.loc[row_index, column_name])) and not (specialTypeInput == SpecialType.MISSING):
                                return False
        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                # Instead of iterating over the columns, we iterate over the rows to check the derived type
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return False
                        elif missing_values is None:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
                        else:
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
            else:  # It works for invalid values and outliers
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
                                    dataDictionary_out.loc[row_index, column_index])) and not (specialTypeInput == SpecialType.MISSING):
                                return False
        elif axis_param is None:
            most_frequent = dataDictionary_in.stack().value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return False
                        elif missing_values is None:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
                        else:
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
            else:  # It works for invalid values and outliers
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                    dataDictionary_out.loc[row_index, column_index])) and not (specialTypeInput == SpecialType.MISSING):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            most_frequent = dataDictionary_in[field].value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for idx, value in dataDictionary_in[field].items():
                    if pd.isnull(value):
                        if dataDictionary_out.loc[idx, field] != most_frequent:
                            return False
                    elif missing_values is None:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            return False
                    else:
                        if value in missing_values:
                            if dataDictionary_out.loc[idx, field] != most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[
                                idx, field]:
                                return False
            else:  # It works for invalid values and outliers
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        if dataDictionary_out.loc[idx, field] != most_frequent:
                            return False
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[
                            idx, field] and not (
                                pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
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
                        if pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_name] != most_frequent:
                                return True
                        elif missing_values is None:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False
                        else:
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, column_name] != most_frequent:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return False
            else:  # It works for invalid values and outliers
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    most_frequent = dataDictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_name] != most_frequent:
                                return True
                        else:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_name])) and not (specialTypeInput == SpecialType.MISSING):
                                return False
        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                # Instead of iterating over the columns, we iterate over the rows to check the derived type
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return True
                        elif missing_values is None:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
                        else:
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
            else:  # It works for invalid values and outliers
                for row_index, row in dataDictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return True
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_index])) and not (specialTypeInput == SpecialType.MISSING):
                                return False
        elif axis_param is None:
            most_frequent = dataDictionary_in.stack().value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if pd.isnull(value):
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return True
                        elif missing_values is None:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index]:
                                return False
                        else:
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
            else:  # It works for invalid values and outliers
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values:
                            if dataDictionary_out.loc[row_index, column_index] != most_frequent:
                                return True
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_index])) and not (specialTypeInput == SpecialType.MISSING):
                                return False

    elif field is not None:
        if field not in dataDictionary_in.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_in.columns:
            most_frequent = dataDictionary_in[field].value_counts().idxmax()
            if specialTypeInput == SpecialType.MISSING:
                for idx, value in dataDictionary_in[field].items():
                    if pd.isnull(value):
                        if dataDictionary_out.loc[idx, field] != most_frequent:
                            return True
                    elif missing_values is None:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                            return False
                    else:
                        if value in missing_values:
                            if dataDictionary_out.loc[idx, field] != most_frequent:
                                return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[
                                idx, field]:
                                return False
            else:  # It works for invalid values and outliers
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        if dataDictionary_out.loc[idx, field] != most_frequent:
                            return True
                    else:
                        if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[
                            idx, field] and not (
                                pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                            dataDictionary_out.loc[idx, field])):
                            return False

    return False



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
            else:  # It works for invalid values and outliers
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
            else:  # It works for invalid values and outliers
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
            else:  # It works for invalid values and outliers
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
            else:  # It works for invalid values and outliers
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
            else:  # It works for invalid values and outliers
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
            else:  # It works for invalid values and outliers
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
            else:  # It works for invalid values and outliers
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
            else:  # It works for invalid values and outliers
                for idx, value in dataDictionary_in[field].items():
                    if value in missing_values:
                        return False

    return True


def checkSpecialTypeMostFrequent(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame, specialTypeInput: SpecialType,
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
    Check if the derived type previous value is applied correctly when the input and output dataframes when belongOp_in and belongOp_out are both BELONG

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
                                    row_index - 1, column_name] and (
                                    not pd.isnull(dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                    dataDictionary_out.loc[row_index - 1, column_name])):
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
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
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            if column_index == 0:
                                if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                    row_index, column_index]:
                                    return False
                            else:
                                if column_index - 1 in dataDictionary_in.columns:
                                    if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                        row_index, column_index - 1]:
                                        return False
                        else:
                            if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                row_index, column_index]:
                                return False

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.iloc[row_index, column_index]
                        if value in missing_values:
                            if column_index == 0:
                                if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if column_index - 1 in dataDictionary_in.columns:
                                    if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                        row_index, column_index - 1]:
                                        return False
                        else:
                            if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.iloc[row_index, column_index]) and pd.isnull(
                                    dataDictionary_out.iloc[row_index, column_index])):
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
                        if pd.isnull(value):
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
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field] and not pd.isnull(dataDictionary_in.loc[idx - 1, field]):
                                    return False
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False

    return True


def checkDerivedTypePreviousBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                            specialTypeInput: SpecialType, missing_values: list, axis_param: int,
                                            field: str) -> bool:
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
                                    return True
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index - 1, column_name]:
                                    return True
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == 0:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return True
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index - 1, column_name]:
                                    return True
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_name])):
                                return False


        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            if column_index == 0:
                                if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                    row_index, column_index]:
                                    return True
                            else:
                                if column_index - 1 in dataDictionary_in.columns:
                                    if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                        row_index, column_index - 1]:
                                        return True
                        else:
                            if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                row_index, column_index]:
                                return False


            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.iloc[row_index, column_index]
                        if value in missing_values:
                            if column_index == 0:
                                if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                    row_index, column_index]:
                                    return True
                            else:
                                if column_index - 1 in dataDictionary_in.columns:
                                    if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                        row_index, column_index - 1]:
                                        return True
                        else:
                            if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.iloc[row_index, column_index]) and pd.isnull(
                                    dataDictionary_out.iloc[row_index, column_index])):
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
                                    return True
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field]:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False
                else:  # missing_values is None
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == 0:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return True
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field]:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == 0:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return True
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx - 1, field]:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False

    return False


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

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            return False

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.iloc[row_index, column_index]
                        if value in missing_values:
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

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            return False

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.iloc[row_index, column_index]
                        if value in missing_values:
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


def checkSpecialTypePrevious(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                             specialTypeInput: SpecialType,
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
        result = checkDerivedTypePreviousBelongBelong(dataDictionary_in=dataDictionary_in,
                                                      dataDictionary_out=dataDictionary_out,
                                                      specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                      axis_param=axis_param, field=field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result = checkDerivedTypePreviousBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                         dataDictionary_out=dataDictionary_out,
                                                         specialTypeInput=specialTypeInput,
                                                         missing_values=missing_values,
                                                         axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkDerivedTypePreviousNotBelongBelong(dataDictionary_in=dataDictionary_in,
                                                         dataDictionary_out=dataDictionary_out,
                                                         specialTypeInput=specialTypeInput,
                                                         missing_values=missing_values,
                                                         axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkDerivedTypePreviousNotBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                            dataDictionary_out=dataDictionary_out,
                                                            specialTypeInput=specialTypeInput,
                                                            missing_values=missing_values, axis_param=axis_param,
                                                            field=field)

    return True if result else False


def checkDerivedTypeNextBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                     specialTypeInput: SpecialType, missing_values: list, axis_param: int,
                                     field: str) -> bool:
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

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
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
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            if column_index == len(row) - 1:
                                if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                    row_index, column_index + 1]:
                                    return False
                        else:
                            if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                row_index, column_index]:
                                return False

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index, value in row.items():
                        column_names = dataDictionary_in.columns.tolist()
                        current_column_index = column_names.index(column_index)
                        if current_column_index < len(column_names) - 1:  # check if it's not the last column
                            next_column_name = column_names[current_column_index + 1]
                        if value in missing_values:
                            if column_index == len(row) - 1:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, column_index]:
                                    return False
                            else:
                                if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                    row_index, next_column_name]:
                                    return False
                        else:
                            if dataDictionary_out.loc[row_index, column_index] != dataDictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.loc[row_index, column_index]) and pd.isnull(
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
                        if pd.isnull(value):
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
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False

    return True


def checkDerivedTypeNextBelongNotBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        specialTypeInput: SpecialType, missing_values: list, axis_param: int,
                                        field: str) -> bool:
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
                                    return True
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index + 1, column_name]:
                                    return True
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name]:
                                return False

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            if row_index == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index, column_name]:
                                    return True
                            else:
                                if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                    row_index + 1, column_name]:
                                    return True
                        else:
                            if dataDictionary_out.loc[row_index, column_name] != dataDictionary_in.loc[
                                row_index, column_name] and not (pd.isnull(
                                dataDictionary_in.loc[row_index, column_name]) and pd.isnull(
                                dataDictionary_out.loc[row_index, column_name])):
                                return False


        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            if column_index == len(row) - 1:
                                if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                    row_index, column_index]:
                                    return True
                            else:
                                if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                    row_index, column_index + 1]:
                                    return True
                        else:
                            if dataDictionary_out.at[row_index, column_index] != dataDictionary_in.at[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.at[row_index, column_index]) and pd.isnull(
                                    dataDictionary_out.at[row_index, column_index])):
                                return False

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.iloc[row_index, column_index]
                        if value in missing_values:
                            if column_index == len(row) - 1:
                                if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                    row_index, column_index]:
                                    return True
                            else:
                                if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                    row_index, column_index + 1]:
                                    return True
                        else:
                            if dataDictionary_out.iloc[row_index, column_index] != dataDictionary_in.iloc[
                                row_index, column_index] and not (
                                    pd.isnull(dataDictionary_in.iloc[row_index, column_index]) and pd.isnull(
                                    dataDictionary_out.iloc[row_index, column_index])):
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
                                    return True
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx + 1, field]:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

                else:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return True
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx + 1, field]:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                return False

            elif specialTypeInput == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in dataDictionary_in[field].items():
                        if value in missing_values:
                            if idx == len(dataDictionary_in) - 1:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field]:
                                    return True
                            else:
                                if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx + 1, field]:
                                    return True
                        else:
                            if dataDictionary_out.loc[idx, field] != dataDictionary_in.loc[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False

    return False


def checkDerivedTypeNextNotBelongBelong(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                        specialTypeInput: SpecialType, missing_values: list, axis_param: int,
                                        field: str) -> bool:
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

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            return False
            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.iloc[row_index, column_index]
                        if value in missing_values:
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
                                           specialTypeInput: SpecialType, missing_values: list, axis_param: int,
                                           field: str) -> bool:
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

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if value in missing_values:
                            return False

        elif axis_param == 1:
            if specialTypeInput == SpecialType.MISSING:
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            return False
            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in dataDictionary_in.iterrows():
                    for column_index in range(len(dataDictionary_in.columns)):
                        value = dataDictionary_in.iloc[row_index, column_index]
                        if value in missing_values:
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


def checkSpecialTypeNext(dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                         specialTypeInput: SpecialType,
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
        result = checkDerivedTypeNextBelongBelong(dataDictionary_in=dataDictionary_in,
                                                  dataDictionary_out=dataDictionary_out,
                                                  specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                  axis_param=axis_param, field=field)
    elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
        result = checkDerivedTypeNextBelongNotBelong(dataDictionary_in=dataDictionary_in,
                                                     dataDictionary_out=dataDictionary_out,
                                                     specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                     axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
        result = checkDerivedTypeNextNotBelongBelong(dataDictionary_in=dataDictionary_in,
                                                     dataDictionary_out=dataDictionary_out,
                                                     specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                     axis_param=axis_param, field=field)
    elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
        result = checkDerivedTypeNextNotBelongNotBelong(dataDictionary_in=dataDictionary_in,
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
                                    if dataDictionary_out.at[idx, col] != dataDictionary_in[
                                        col].value_counts().idxmax():
                                        return False
                                else:
                                    if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                        return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.loc[idx].mode()[0]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False

            elif derivedTypeOutput == DerivedType.PREVIOUS:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != 0:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx - 1, col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != 0:
                                prev_col = row.index[row.index.get_loc(col) - 1]  # Get the previous column
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, prev_col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False

            elif derivedTypeOutput == DerivedType.NEXT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != len(dataDictionary_in) - 1:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx + 1, col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False
                elif axis_param == 1:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != dataDictionary_in.columns[-1]:
                                next_col = dataDictionary_in.columns[dataDictionary_in.columns.get_loc(col) + 1]
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, next_col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False

        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
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
                                    if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                            pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                        dataDictionary_out.loc[idx, col])):
                                        return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1:
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.loc[idx].mode()[0]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False

            elif derivedTypeOutput == DerivedType.PREVIOUS:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != 0:
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.at[idx - 1, col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False
                elif axis_param == 1:
                    for idx, row in dataDictionary_in.iterrows():
                        for col in row.index:
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != 0:
                                prev_col = row.index[row.index.get_loc(col) - 1]  # Get the previous column
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.at[idx, prev_col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False

            elif derivedTypeOutput == DerivedType.NEXT:
                if axis_param == 0:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and idx != len(dataDictionary_in) - 1:
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.at[idx + 1, col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
                                    return False
                elif axis_param == 1:
                    for col in dataDictionary_in.columns:
                        for idx, value in dataDictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1 and col != dataDictionary_in.columns[-1]:
                                next_col = dataDictionary_in.columns[dataDictionary_in.columns.get_loc(col) + 1]
                                if dataDictionary_out.at[idx, col] == dataDictionary_in.at[idx, next_col]:
                                    return False
                            else:
                                if dataDictionary_out.at[idx, col] != dataDictionary_in.at[idx, col] and not (
                                        pd.isnull(dataDictionary_in.loc[idx, col]) and pd.isnull(
                                    dataDictionary_out.loc[idx, col])):
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
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != 0:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx - 1, field]:
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False
                elif derivedTypeOutput == DerivedType.NEXT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != len(dataDictionary_in) - 1:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx + 1, field]:
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False

            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1:
                            if dataDictionary_out.at[idx, field] == dataDictionary_in[field].value_counts().idxmax():
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != 0:
                            if dataDictionary_out.at[idx, field] == dataDictionary_in.at[idx - 1, field]:
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
                                return False
                elif derivedTypeOutput == DerivedType.NEXT:
                    for idx, value in dataDictionary_in[field].items():
                        if outliers_dataframe_mask.at[idx, field] == 1 and idx != len(dataDictionary_in) - 1:
                            if dataDictionary_out.at[idx, field] == dataDictionary_in.at[idx + 1, field]:
                                return False
                        else:
                            if dataDictionary_out.at[idx, field] != dataDictionary_in.at[idx, field] and not (
                                    pd.isnull(dataDictionary_in.loc[idx, field]) and pd.isnull(
                                dataDictionary_out.loc[idx, field])):
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
