# Importing libraries
import math

import numpy as np
import pandas as pd

# Importing enumerations from packages
from helpers.auxiliar import find_closest_value, check_interval_condition, outlier_closest
from helpers.enumerations import DerivedType, Belong, SpecialType, Closure
from helpers.logger import print_and_log


def check_fix_value_most_frequent(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                  fix_value_input, belong_op_out: Belong, axis_param: int, field_in: str, field_out: str) -> bool:
    """
    Check if the most frequent value is applied correctly on the fix input value
    to the data_dictionary_out respect to the data_dictionary_in when belong_op_in is always BELONG

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
        :param fix_value_input: input value to apply the most frequent value
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the most frequent value
        :param field_in: (str) field to apply the most frequent value
        :param field_out: (str) field to apply the most frequent value

    Returns:
        :return: True if the most frequent value is applied correctly on the fix input value
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in data_dictionary_in.iterrows():
                most_frequent_value = row.value_counts().idxmax()
                for column_index, value in row.items():
                    if value == fix_value_input:
                        if data_dictionary_out.at[row_index, column_index] != most_frequent_value:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[row_index, column_index]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {row_index} and column: {column_index} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[row_index, column_index]}")
                    else:
                        if data_dictionary_out.at[row_index, column_index] != data_dictionary_in.at[row_index, column_index]:
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
        elif axis_param == 0:  # Applies the lambda function at the column level
            for col in data_dictionary_in.columns:
                most_frequent_value = data_dictionary_in[col].value_counts().idxmax()
                for idx, value in data_dictionary_in[col].items():
                    if value == fix_value_input:
                        if data_dictionary_out.at[idx, col] != most_frequent_value and not (
                                pd.isnull(data_dictionary_out.at[idx, col]) or pd.isnull(
                            data_dictionary_out.at[idx, col])):
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, col]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, col]}")
                    else:
                        if data_dictionary_out.loc[idx, col] != data_dictionary_in.loc[idx, col]:
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.loc[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")
        else:  # Applies at the dataframe level
            # Calculate the most frequent value
            most_frequent_value = data_dictionary_in.stack().value_counts().idxmax()
            for col_name in data_dictionary_in.columns:
                for idx, value in data_dictionary_in[col_name].items():
                    if value == fix_value_input:
                        if data_dictionary_out.at[idx, col_name] != most_frequent_value:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name] and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                            data_dictionary_in.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.loc[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            most_frequent_value = data_dictionary_in[field_in].value_counts().idxmax()
            for idx, value in data_dictionary_in[field_in].items():
                if value == fix_value_input:
                    if data_dictionary_out.at[idx, field_out] != most_frequent_value and not (
                            pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(
                                    data_dictionary_out.at[idx, field_out])):
                        if belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, field_out]}")
                else:
                    if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                            pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(
                                    data_dictionary_out.at[idx, field_out])):
                        keep_no_trans_result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_fix_value_previous(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame, fix_value_input,
                             belong_op_out: Belong, axis_param: int, field_in: str, field_out: str) -> bool:
    """
    Check if the previous value is applied correctly to the fix input value
    to the data_dictionary_out respect to the data_dictionary_in when belong_op_in is always BELONG

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the previous value
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the previous value
        :param fix_value_input: input value to apply the previous value
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the previous value
        :param field_in: (str) field to apply the previous value
        :param field_out: (str) field to apply the previous value

    Returns:
        :return: True if the previous value is applied correctly to the fix input value
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in data_dictionary_in.iterrows():
                for column_name, value in row.items():
                    column_index = data_dictionary_in.columns.get_loc(column_name)
                    value = data_dictionary_in.at[row_index, column_name]
                    if value == fix_value_input:
                        if column_index == 0:
                            if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                                row_index, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                        else:
                            if column_index - 1 in data_dictionary_in.columns:
                                if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                                    row_index, column_index - 1]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index-1]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index-1]} but is: {data_dictionary_out.at[row_index, column_name]}")
                    else:
                        if data_dictionary_out.iloc[row_index, column_index] != data_dictionary_in.iloc[
                            row_index, column_index]:
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")

        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(data_dictionary_in.columns):
                for row_index, value in data_dictionary_in[column_name].items():
                    if value == fix_value_input:
                        if row_index == 0:
                            if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                                row_index, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index - 1, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index-1, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index-1, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                    else:
                        if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                            row_index, column_name] and not (
                                pd.isnull(data_dictionary_in.loc[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.loc[row_index - 1, column_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            for idx, value in data_dictionary_in[field_in].items():
                if value == fix_value_input:
                    if idx == 0:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                    else:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx - 1, field_in]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx-1, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx-1, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                else:
                    if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                        if idx != 0 and (
                                not pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(
                                data_dictionary_out.loc[idx - 1, field_out])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_fix_value_next(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame, fix_value_input,
                         belong_op_out: Belong, axis_param: int, field_in: str, field_out: str) -> bool:
    """
    Check if the next value is applied correctly to the fix input value
    to the data_dictionary_out respect to the data_dictionary_in when belong_op_in is always BELONG

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the next value
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the next value
        :param fix_value_input: input value to apply the next value
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the next value
        :param field_in: (str) field to apply the next value
        :param field_out: (str) field to apply the next value

    Returns:
        :return: True if the next value is applied correctly to the fix input value
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in data_dictionary_in.iterrows():
                for column_name, value in row.items():
                    column_index = data_dictionary_in.columns.get_loc(column_name)
                    value = data_dictionary_in.at[row_index, column_name]
                    if value == fix_value_input:
                        if column_index == len(row) - 1:
                            if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                                row_index, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                        else:
                            if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.iloc[
                                row_index, column_index + 1]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index + 1]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index + 1]} but is: {data_dictionary_out.at[row_index, column_name]}")
                    else:
                        if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                            row_index, column_name] and (
                                not pd.isnull(data_dictionary_in.at[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(data_dictionary_in.columns):
                for row_index, value in data_dictionary_in[column_name].items():
                    if value == fix_value_input:
                        if row_index == len(data_dictionary_in) - 1:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index + 1, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index+1, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index+1, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                    else:
                        if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                            row_index, column_name] and (
                                not pd.isnull(data_dictionary_in.at[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")

        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            for idx, value in data_dictionary_in[field_in].items():
                if value == fix_value_input:
                    if idx == len(data_dictionary_in) - 1:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                    else:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx + 1, field_in]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx + 1, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx + 1, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                else:
                    if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and (
                            not pd.isnull(data_dictionary_in.at[idx, field_in]) and pd.isnull(
                                data_dictionary_out.at[idx, field_out])):
                        keep_no_trans_result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_interval_most_frequent(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                 left_margin: float, right_margin: float, closure_type: Closure,
                                 belong_op_out: Belong,
                                 axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the most frequent value is applied correctly on the interval
    to the data_dictionary_out respect to the data_dictionary_in when belong_op_in is always BELONG

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the most frequent value
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the most frequent value
        :param left_margin: (float) left margin of the interval
        :param right_margin: (float) right margin of the interval
        :param closure_type: (Closure) closure of the interval
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the most frequent value
        :param field_in: (str) field to apply the most frequent value
        :param field_out: (str) field to apply the most frequent value

    Returns:
        :return: True if the most frequent value is applied correctly on the interval
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in data_dictionary_in.iterrows():
                most_frequent_value = row.value_counts().idxmax()
                for column_index, value in row.items():
                    if check_interval_condition(value, left_margin, right_margin, closure_type):
                        if data_dictionary_out.at[row_index, column_index] != most_frequent_value:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[row_index, column_index]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {row_index} and column: {column_index} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[row_index, column_index]}")
                    else:
                        if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[row_index, column_index] and (
                                not pd.isnull(data_dictionary_in.loc[row_index, column_index]) and pd.isnull(
                                data_dictionary_out.loc[row_index, column_index])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
        elif axis_param == 0:  # Applies the lambda function at the column level
            for col in data_dictionary_in.columns:
                most_frequent_value = data_dictionary_in[col].value_counts().idxmax()
                for idx, value in data_dictionary_in[col].items():
                    if check_interval_condition(value, left_margin, right_margin, closure_type):
                        if data_dictionary_out.at[idx, col] != most_frequent_value:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, col]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, col]}")
                    else:
                        if data_dictionary_out.loc[idx, col] != data_dictionary_in.loc[idx, col] and (
                                not pd.isnull(data_dictionary_in.at[idx, col]) and pd.isnull(
                                data_dictionary_out.at[idx, col])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.loc[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")
        else:  # Applies at the dataframe level
            # Calculate the most frequent value
            most_frequent_value = data_dictionary_in.stack().value_counts().idxmax()
            for col_name in data_dictionary_in.columns:
                for idx, value in data_dictionary_in[col_name].items():
                    if check_interval_condition(value, left_margin, right_margin, closure_type):
                        if data_dictionary_out.at[idx, col_name] != most_frequent_value:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name] and (
                                not pd.isnull(data_dictionary_in.at[idx, col_name]) and pd.isnull(
                                data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.loc[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            most_frequent_value = data_dictionary_in[field_in].value_counts().idxmax()
            for idx, value in data_dictionary_in[field_in].items():
                if check_interval_condition(value, left_margin, right_margin, closure_type):
                    if data_dictionary_out.at[idx, field_out] != most_frequent_value:
                        if belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {most_frequent_value} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_out == Belong.NOTBELONG:
                            result = True
                else:
                    if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and (
                            not pd.isnull(data_dictionary_in.at[idx, field_in]) and pd.isnull(
                                data_dictionary_out.at[idx, field_out])):
                        keep_no_trans_result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_interval_previous(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                            left_margin: float, right_margin: float, closure_type: Closure,
                            belong_op_out: Belong = Belong.BELONG,
                            axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the previous value is applied correctly on the interval
    to the data_dictionary_out respect to the data_dictionary_in when belong_op_in is always BELONG

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the previous value
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the previous value
        :param left_margin: (float) left margin of the interval
        :param right_margin: (float) right margin of the interval
        :param closure_type: (Closure) closure of the interval
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the previous value
        :param field_in: (str) field to apply the previous value
        :param field_out: (str) field to apply the previous value

    Returns:
        :return: True if the previous value is applied correctly on the interval
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in data_dictionary_in.iterrows():
                for column_name, value in row.items():
                    column_index = data_dictionary_in.columns.get_loc(column_name)
                    value = data_dictionary_in.at[row_index, column_name]
                    if check_interval_condition(value, left_margin, right_margin, closure_type):
                        if column_index == 0:
                            if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                                row_index, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                        else:
                            if column_index - 1 in data_dictionary_in.columns:
                                if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                                    row_index, column_index - 1] and (
                                    not pd.isnull(data_dictionary_in.at[
                                    row_index, column_index - 1]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index-1]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index-1]} but is: {data_dictionary_out.at[row_index, column_name]}")
                    else:
                        if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                            row_index, column_name] and (
                                    not pd.isnull(data_dictionary_in.at[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(data_dictionary_in.columns):
                for row_index, value in data_dictionary_in[column_name].items():
                    if check_interval_condition(value, left_margin, right_margin, closure_type):
                        if row_index == 0:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index - 1, column_name] and (
                                    not pd.isnull(data_dictionary_in.loc[
                                row_index - 1, column_name]) and pd.isnull(
                                data_dictionary_out.loc[row_index, column_name])):
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index-1, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index-1, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                    else:
                        if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                            row_index, column_name] and (
                                    not pd.isnull(data_dictionary_in.at[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            for idx, value in data_dictionary_in[field_in].items():
                if check_interval_condition(value, left_margin, right_margin, closure_type):
                    if idx == 0:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                    else:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx - 1, field_in] and (
                                    not pd.isnull(data_dictionary_in.loc[idx - 1, field_in]) and pd.isnull(
                                data_dictionary_out.loc[idx, field_out])):
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx-1, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx-1, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                else:
                    if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and (
                                    not pd.isnull(data_dictionary_in.at[idx, field_in]) and pd.isnull(
                                data_dictionary_out.at[idx, field_out])):
                        keep_no_trans_result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_interval_next(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                        left_margin: float, right_margin: float, closure_type: Closure,
                        belong_op_out: Belong = Belong.BELONG,
                        axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the next value is applied correctly on the interval
    to the data_dictionary_out respect to the data_dictionary_in when belong_op_in is always BELONG

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the next value
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the next value
        :param left_margin: (float) left margin of the interval
        :param right_margin: (float) right margin of the interval
        :param closure_type: (Closure) closure of the interval
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the next value
        :param field_in: (str) field to apply the next value
        :param field_out: (str) field to apply the next value

    Returns:
        :return: True if the next value is applied correctly on the interval
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param == 1:  # Applies in a row level
            for row_index, row in data_dictionary_in.iterrows():
                for column_name, value in row.items():
                    column_index = data_dictionary_in.columns.get_loc(column_name)
                    value = data_dictionary_in.at[row_index, column_name]
                    if check_interval_condition(value, left_margin, right_margin, closure_type):
                        if column_index == len(row) - 1:
                            if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                                row_index, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_name]} but is: {data_dictionary_out.at[row_index, column_name]}")
                        else:
                            if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.iloc[
                                row_index, column_index + 1] and (
                                    not pd.isnull(data_dictionary_in.iloc[
                                row_index, column_index + 1]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index + 1]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index + 1]} but is: {data_dictionary_out.at[row_index, column_name]}")
                    else:
                        if data_dictionary_out.at[row_index, column_name] != data_dictionary_in.at[
                            row_index, column_name] and (
                                    not pd.isnull(data_dictionary_in.at[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
        elif axis_param == 0:  # Applies at the column level
            for column_index, column_name in enumerate(data_dictionary_in.columns):
                for row_index, value in data_dictionary_in[column_name].items():
                    if check_interval_condition(value, left_margin, right_margin, closure_type):
                        if row_index == len(data_dictionary_in) - 1:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index + 1, column_name] and (
                                    not pd.isnull(data_dictionary_in.at[row_index + 1, column_name]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index + 1, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index + 1, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
                    else:
                        if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                            row_index, column_name] and not (
                                     pd.isnull(data_dictionary_in.at[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_name]}")
        else:  # Applies at the dataframe level
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            for idx, value in data_dictionary_in[field_in].items():
                if check_interval_condition(value, left_margin, right_margin, closure_type):
                    if idx == len(data_dictionary_in) - 1:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                    else:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx + 1, field_in] and (
                                     not pd.isnull(data_dictionary_in.loc[idx + 1, field_in]) and pd.isnull(
                                data_dictionary_out.loc[idx, field_out])):
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx + 1, field_out]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx + 1, field_out]} but is: {data_dictionary_out.at[idx, field_out]}")
                else:
                    if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                                     pd.isnull(data_dictionary_in.at[idx, field_in]) and pd.isnull(
                                data_dictionary_out.at[idx, field_out])):
                        keep_no_trans_result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.iloc[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_fix_value_interpolation(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                  fix_value_input, belong_op_out: Belong, axis_param: int = None,
                                  field_in: str = None, field_out: str = None) -> bool:
    """
        Check if the interpolation is applied correctly to the fix value input when the input and output dataframe
        when belong_op_in is BELONG and belong_op_out is BELONG or NOTBELONG
        params:
            :param data_dictionary_in: dataframe with the data before the interpolation
            :param data_dictionary_out: dataframe with the data after the interpolation
            :param fix_value_input: value to apply the interpolation
            :param belong_op_out: then condition to check the invariant
            :param axis_param: axis to apply the interpolation
            :param field_in: field to apply the interpolation
            :param field_out: field to apply the interpolation

        Returns:
            :return: True if the interpolation is applied correctly to the fix value input
    """

    if axis_param is None and field_in is None:
        raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    data_dictionary_in_copy = data_dictionary_in.copy()
    if field_in is None:
        if axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                data_dictionary_in_copy[col_name] = (
                    data_dictionary_in[col_name].apply(lambda x: np.nan if x == fix_value_input else x).
                    interpolate(method='linear', limit_direction='both'))

            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                for idx in data_dictionary_in.index:
                    if data_dictionary_in.at[idx, col_name] == fix_value_input:
                        if data_dictionary_out.at[idx, col_name] != data_dictionary_in_copy.at[idx, col_name]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.at[idx, col_name] != data_dictionary_in.at[idx, col_name]) and not(pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 1:
            for idx, row in data_dictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                data_dictionary_in_copy[row] = (
                    numeric_data[row].apply(lambda x: np.nan if x == fix_value_input else x).
                    interpolate(method='linear', limit_direction='both'))
            for col_name in data_dictionary_in.columns:
                for idx in data_dictionary_in.index:
                    if data_dictionary_in.at[idx, col_name] == fix_value_input:
                        if data_dictionary_out.at[idx, col_name] != data_dictionary_in_copy.at[idx, col_name]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.at[idx, col_name] != data_dictionary_in.at[idx, col_name]) and not(pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("The field is not numeric")

        data_dictionary_in_copy[field_in] = (data_dictionary_in[field_in].apply(lambda x: np.nan if x == fix_value_input else x).
                                         interpolate(method='linear', limit_direction='both'))

        for idx in data_dictionary_in.index:
            if data_dictionary_in.at[idx, field_in] == fix_value_input:
                if data_dictionary_out.at[idx, field_out] != data_dictionary_in_copy.at[idx, field_in]:
                    if belong_op_out == Belong.BELONG:
                        result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                    elif belong_op_out == Belong.NOTBELONG:
                        result = True
                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
            else:
                if (data_dictionary_out.at[idx, field_out] != data_dictionary_in.at[idx, field_in]) and not(pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(data_dictionary_out.at[idx, field_out])):
                    keep_no_trans_result = False
                    print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_fix_value_mean(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame, fix_value_input,
                         belong_op_out: Belong, axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the special type mean is applied correctly when the input and output dataframes
    when belong_op_in and belong_op_out are BELONG
    params:
        :param data_dictionary_in: dataframe with the data before the mean
        :param data_dictionary_out: dataframe with the data after the mean
        :param fix_value_input: fix value to apply the mean
        :param belong_op_out: then condition to check the invariant
        :param axis_param: axis to apply the mean
        :param field_in: field to apply the mean
        :param field_out: field to apply the mean

    Returns:
        :return: True if the special type mean is applied correctly
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
            # Calculate the mean of these numeric columns
            mean_value = only_numbers_df.mean().mean()
            # Check the data_dictionary_out positions with missing values have been replaced with the mean
            for col_name in data_dictionary_in.columns:
                for idx, value in data_dictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if data_dictionary_in.at[idx, col_name] == fix_value_input:
                            if data_dictionary_out.at[idx, col_name] != mean_value:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                        else:
                            if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                    data_dictionary_out.at[idx, col_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the data_dictionary_out positions with missing values have been replaced with the mean
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                mean = data_dictionary_in[col_name].mean()
                for idx, value in data_dictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if data_dictionary_in.at[idx, col_name] == fix_value_input:
                            if data_dictionary_out.at[idx, col_name] != mean:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean} but is: {data_dictionary_out.loc[idx, col_name]}")
                        else:
                            if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                    data_dictionary_out.at[idx, col_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 1:
            for idx, row in data_dictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                mean = numeric_data.mean()
                # Check if the missing values in the row have been replaced with the mean in data_dictionary_out
                for col_name, value in numeric_data.items():
                    if value == fix_value_input:
                        if data_dictionary_out.at[idx, col_name] != mean:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the data_dictionary_out positions with missing values have been replaced with the mean
        mean = data_dictionary_in[field_in].mean()
        for idx, value in data_dictionary_in[field_in].items():
            if value == fix_value_input:
                if data_dictionary_out.at[idx, field_out] != mean:
                    if belong_op_out == Belong.BELONG:
                        result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {mean} but is: {data_dictionary_out.loc[idx, field_out]}")
                    elif belong_op_out == Belong.NOTBELONG:
                        result = True
                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {mean} but is: {data_dictionary_out.loc[idx, field_out]}")
            else:
                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                                pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(
                                data_dictionary_out.at[idx, field_out])):
                    keep_no_trans_result = False
                    print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_fix_value_median(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame, fix_value_input,
                           belong_op_out: Belong, axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the median is applied correctly to the fix input value when the input and output dataframes
    when belong_op_in and belong_op_out are BELONG
    params:
        :param data_dictionary_in: dataframe with the data before the median
        :param data_dictionary_out: dataframe with the data after the median
        :param fix_value_input: fix value to apply the median
        :param belong_op_out: then condition to check the invariant
        :param axis_param: axis to apply the median
        :param field_in: field to apply the median
        :param field_out: field to apply the median

    Returns:
        :return: True if the median is applied correctly to the fix value
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
            # Calculate the median of these numeric columns
            median_value = only_numbers_df.median().median()
            # Check the data_dictionary_out positions with missing values have been replaced with the median
            for col_name in data_dictionary_in.columns:
                for idx, value in data_dictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if data_dictionary_in.at[idx, col_name] == fix_value_input:
                            if data_dictionary_out.at[idx, col_name] != median_value:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                        else:
                            if data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name] and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                data_dictionary_out.at[idx, col_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the data_dictionary_out positions with missing values have been replaced with the median
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                median = data_dictionary_in[col_name].median()
                for idx, value in data_dictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if data_dictionary_in.at[idx, col_name] == fix_value_input:
                            if data_dictionary_out.at[idx, col_name] != median:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                        else:
                            if data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name] and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                data_dictionary_out.at[idx, col_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 1:
            for idx, row in data_dictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                median = numeric_data.median()
                # Check if the missing values in the row have been replaced with the median in data_dictionary_out
                for col_name, value in numeric_data.items():
                    if value == fix_value_input:
                        if data_dictionary_out.at[idx, col_name] != median:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name]:
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")

        # Check the data_dictionary_out positions with missing values have been replaced with the median
        median = data_dictionary_in[field_in].median()
        for idx, value in data_dictionary_in[field_in].items():
            if value == fix_value_input:
                if data_dictionary_out.at[idx, field_out] != median:
                    if belong_op_out == Belong.BELONG:
                        result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")
                    elif belong_op_out == Belong.NOTBELONG:
                        result = True
                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")
            else:
                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                                pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(
                                data_dictionary_out.at[idx, field_out])):
                    keep_no_trans_result = False
                    print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_fix_value_closest(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame, fix_value_input,
                            belong_op_out: Belong, axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
        Check if the closest is applied correctly to the fix input value
        when the input and output dataframes when belong_op_in is Belong
        params:
            :param data_dictionary_in: dataframe with the data before the closest
            :param data_dictionary_out: dataframe with the data after the closest
            :param fix_value_input: fix value to apply the closest
            :param belong_op_out: then condition to check the invariant
            :param axis_param: axis to apply the closest
            :param field_in: field to apply the closest
            :param field_out: field to apply the closest

        Returns:
            :return: True if the closest is applied correctly to the fix input value
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param is None:
            only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
            # Flatten the DataFrame into a single series of values
            flattened_values = only_numbers_df.values.flatten().tolist()
            # Find the closest numeric value to the fix value
            closest_value = find_closest_value(flattened_values, fix_value_input)
            # Replace the missing values with the closest numeric values
            for i in range(len(data_dictionary_in.index)):
                for j in range(len(data_dictionary_in.columns)):
                    current_value = data_dictionary_in.iloc[i, j]
                    if current_value == fix_value_input:
                        if data_dictionary_out.iloc[i, j] != closest_value:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {i} and column: {j} value should be: {closest_value} but is: {data_dictionary_out.loc[i, j]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {i} and column: {j} value should be: {closest_value} but is: {data_dictionary_out.loc[i, j]}")
                    else:
                        if pd.isnull(data_dictionary_in.iloc[i, j]):
                            raise ValueError(
                                "Error: it's not possible to apply the closest operation to the null values")
                        if (data_dictionary_out.loc[i, j] != data_dictionary_in.loc[i, j]) and not (
                                pd.isnull(data_dictionary_in.loc[i, j]) or pd.isnull(data_dictionary_out.loc[i, j])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {i} and column: {j} value should be: {data_dictionary_in.loc[i, j]} but is: {data_dictionary_out.loc[i, j]}")
        elif axis_param == 0:
            # Iterate over each column
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                # Flatten the column into a list of values
                flattened_values = data_dictionary_in[col_name].values.flatten().tolist()
                # Find the closest numeric value to the fix value in the column
                closest_value = find_closest_value(flattened_values, fix_value_input)
                # Replace the missing values with the closest numeric values in the column
                for i in range(len(data_dictionary_in.index)):
                    current_value = data_dictionary_in[col_name].iloc[i]
                    if current_value == fix_value_input:
                        if data_dictionary_out[col_name].iloc[i] != closest_value:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {i} and column: {col_name} value should be: {closest_value} but is: {data_dictionary_out.loc[i, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {i} and column: {col_name} value should be: {closest_value} but is: {data_dictionary_out.loc[i, col_name]}")
                    else:
                        if pd.isnull(data_dictionary_in[col_name].iloc[i]):
                            raise ValueError(
                                "Error: it's not possible to apply the closest operation to the null values")
                        if (data_dictionary_out[col_name].iloc[i] != data_dictionary_in[col_name].iloc[i]) and not (
                                pd.isnull(data_dictionary_in[col_name].iloc[i]) or pd.isnull(
                                data_dictionary_out[col_name].iloc[i])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {i} and column: {col_name} value should be: {data_dictionary_in[col_name].iloc[i]} but is: {data_dictionary_out.loc[i, col_name]}")
        elif axis_param == 1:
            # Iterate over each row
            for row_idx in range(len(data_dictionary_in.index)):
                # Get the numeric values in the current row
                row_df = pd.DataFrame([data_dictionary_in.iloc[row_idx]])
                numeric_values_in_row = row_df.select_dtypes(include=[np.number]).values.tolist()
                # Flatten the row into a list of values
                flattened_values = [val for sublist in numeric_values_in_row for val in sublist]
                # Find the closest numeric value to the fix value in the row
                closest_value = find_closest_value(flattened_values, fix_value_input)
                # Replace the missing values with the closest numeric values in the row
                for col_name in data_dictionary_in.columns:
                    current_value = data_dictionary_in.at[row_idx, col_name]
                    if current_value == fix_value_input:
                        if data_dictionary_out.at[row_idx, col_name] != closest_value:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {row_idx} and column: {col_name} value should be: {closest_value} but is: {data_dictionary_out.loc[row_idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {row_idx} and column: {col_name} value should be: {closest_value} but is: {data_dictionary_out.loc[row_idx, col_name]}")
                    else:
                        if pd.isnull(data_dictionary_in.at[
                                         row_idx, col_name]):
                            raise ValueError(
                                "Error: it's not possible to apply the closest operation to the null values")
                        if (data_dictionary_out.at[row_idx, col_name] != data_dictionary_in.at[
                            row_idx, col_name]) and not (
                                pd.isnull(data_dictionary_in.loc[row_idx, col_name]) or pd.isnull(
                                data_dictionary_out.loc[row_idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {row_idx} and column: {col_name} value should be: {data_dictionary_in.at[row_idx, col_name]} but is: {data_dictionary_out.loc[row_idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")

        # Flatten the column into a list of values
        flattened_values = data_dictionary_in[field_in].values.flatten().tolist()
        # Find the closest numeric value to the fix value in the column
        closest_value = find_closest_value(flattened_values, fix_value_input)
        # Replace the missing values with the closest numeric values in the column
        for i in range(len(data_dictionary_in.index)):
            current_value = data_dictionary_in[field_in].iloc[i]
            if current_value == fix_value_input:
                if data_dictionary_out[field_out].iloc[i] != closest_value:
                    if belong_op_out == Belong.BELONG:
                        result = False
                        print_and_log(f"Error in row: {i} and column: {field_out} value should be: {closest_value} but is: {data_dictionary_out.loc[i, field_out]}")
                    elif belong_op_out == Belong.NOTBELONG:
                        result = True
                        print_and_log(f"Row: {i} and column: {field_out} value should be: {closest_value} but is: {data_dictionary_out.loc[i, field_out]}")
            else:
                if pd.isnull(data_dictionary_in[field_in].iloc[i]):
                    raise ValueError(
                        "Error: it's not possible to apply the closest operation to the null values")
                if (data_dictionary_out[field_out].iloc[i] != data_dictionary_in[field_in].iloc[i]) and not (
                        pd.isnull(data_dictionary_in[field_in].iloc[i]) or pd.isnull(
                        data_dictionary_out[field_out].iloc[i])):
                    keep_no_trans_result = False
                    print_and_log(f"Error in row: {i} and column: {field_out} value should be: {data_dictionary_in[field_in].iloc[i]} but is: {data_dictionary_out.loc[i, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_interval_interpolation(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                 left_margin: float, right_margin: float, closure_type: Closure,
                                 belong_op_in: Belong, belong_op_out: Belong, axis_param: int = None,
                                 field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframes when belong_op_in is BELONG and belong_op_out is NOTBELONG
    params:
        :param data_dictionary_in: dataframe with the data before the interpolation
        :param data_dictionary_out: dataframe with the data after the interpolation
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure of the interval
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param axis_param: axis to apply the interpolation
        :param field_in: field to apply the interpolation
        :param field_out: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
    """
    if axis_param is None and field_in is None:
        raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    data_dictionary_in_copy = data_dictionary_in.copy()
    if field_in is None:
        if axis_param == 0:
            for col in data_dictionary_in.columns:
                if np.issubdtype(data_dictionary_in[col].dtype, np.number):
                    data_dictionary_in_copy[col] = data_dictionary_in_copy[col].apply(
                        lambda x: np.nan if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
                    data_dictionary_in_copy[col] = data_dictionary_in_copy[col].interpolate(method='linear',
                                                                                              limit_direction='both')
            # Iterate over each column
            for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                # For each index in the column
                for idx in data_dictionary_in.index:
                    # Verify if the value is NaN in the original dataframe
                    if pd.isnull(data_dictionary_in.at[idx, col]):
                        # Replace the value with the corresponding one from dataDictionary_copy_copy
                        data_dictionary_in_copy.at[idx, col] = data_dictionary_in.at[idx, col]

            # Iterate over each column
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                # Iterate over each index in the column
                for idx in data_dictionary_in.index:
                    if check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type):
                        if data_dictionary_out.at[idx, col_name] != data_dictionary_in_copy.at[idx, col_name]:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.at[idx, col_name] != data_dictionary_in.at[idx, col_name]) and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                            data_dictionary_in.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 1:
            data_dictionary_in_copy = data_dictionary_in_copy.T
            data_dictionary_in = data_dictionary_in.T
            for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                if np.issubdtype(data_dictionary_in[col].dtype, np.number):
                    data_dictionary_in_copy[col] = data_dictionary_in_copy[col].apply(
                        lambda x: np.nan if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
                    data_dictionary_in_copy[col] = data_dictionary_in_copy[col].interpolate(method='linear',
                                                                                              limit_direction='both')
            # Iterate over each column
            for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                # For each index in the column
                for idx in data_dictionary_in.index:
                    # Verify if the value is NaN in the original dataframe
                    if pd.isnull(data_dictionary_in.at[idx, col]):
                        # Replace the value with the corresponding one from dataDictionary_copy_copy
                        data_dictionary_in_copy.at[idx, col] = data_dictionary_in.at[idx, col]
            data_dictionary_in_copy = data_dictionary_in_copy.T
            data_dictionary_in = data_dictionary_in.T

            for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                for idx in data_dictionary_in.index:
                    if check_interval_condition(data_dictionary_in.at[idx, col], left_margin, right_margin,
                                                closure_type):
                        if data_dictionary_out.at[idx, col] != data_dictionary_in_copy.at[idx, col]:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")
                    else:
                        if (data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, col]) and not (
                                pd.isnull(data_dictionary_out.at[idx, col]) or pd.isnull(
                            data_dictionary_in.at[idx, col])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("The field is not numeric")

        data_dictionary_in_copy[field_in] = data_dictionary_in_copy[field_in].apply(
            lambda x: np.nan if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
        data_dictionary_in_copy[field_in] = data_dictionary_in_copy[field_in].interpolate(method='linear',
                                                                                      limit_direction='both')
        # For each index in the column
        for idx in data_dictionary_in.index:
            # Verify if the value is NaN in the original dataframe
            if pd.isnull(data_dictionary_in.at[idx, field_in]):
                # Replace the value with the corresponding one from dataDictionary_copy_copy
                data_dictionary_in_copy.at[idx, field_in] = data_dictionary_in.at[idx, field_in]

        # For each index in the column
        for idx in data_dictionary_in.index:
            # Verify if the value is NaN in the original dataframe
            if pd.isnull(data_dictionary_in.at[idx, field_in]):
                # Replace the value with the corresponding one from data_dictionary_in
                data_dictionary_in_copy.at[idx, field_in] = data_dictionary_in.at[idx, field_in]

        for idx in data_dictionary_in.index:
            if check_interval_condition(data_dictionary_in.at[idx, field_in], left_margin, right_margin, closure_type):
                if data_dictionary_out.at[idx, field_out] != data_dictionary_in_copy.at[idx, field_in]:
                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                        result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                        result = True
                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
            else:
                if (data_dictionary_out.at[idx, field_out] != data_dictionary_in.at[idx, field_in]) and not (
                        pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(
                    data_dictionary_out.at[idx, field_out])):
                    keep_no_trans_result = False
                    print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_interval_mean(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                        left_margin: float, right_margin: float, closure_type: Closure,
                        belong_op_in: Belong, belong_op_out: Belong, axis_param: int = None,
                        field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the mean is applied correctly on the interval
    to the data_dictionary_out respect to the data_dictionary_in

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the mean
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the mean
        :param left_margin: (float) left margin of the interval
        :param right_margin: (float) right margin of the interval
        :param closure_type: (Closure) closure of the interval
        :param belong_op_in: (Belong) if condition to check the invariant
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the mean
        :param field_in: (str) field to apply the mean
        :param field_out: (str) field to apply the mean

    Returns:
        :return: True if the mean is applied correctly on the interval
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
            # Calculate the mean of these numeric columns
            mean_value = only_numbers_df.mean()
            # Check the data_dictionary_out positions with missing values have been replaced with the mean
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                for idx, value in data_dictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type):
                            if data_dictionary_out.at[idx, col_name] != mean_value:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                        else:
                            if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                    data_dictionary_out.at[idx, col_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the data_dictionary_out positions with missing values have been replaced with the mean
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                mean = data_dictionary_in[col_name].mean()
                for idx, value in data_dictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type):
                            if data_dictionary_out.at[idx, col_name] != mean:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean} but is: {data_dictionary_out.loc[idx, col_name]}")
                        else:
                            if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                    data_dictionary_out.at[idx, col_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 1:
            for idx, row in data_dictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                mean = numeric_data.mean()
                # Check if the missing values in the row have been replaced with the mean in data_dictionary_out
                for col_name, value in numeric_data.items():
                    if check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type):
                        if data_dictionary_out.at[idx, col_name] != mean:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the data_dictionary_out positions with missing values have been replaced with the mean
        mean = data_dictionary_in[field_in].mean()
        for idx, value in data_dictionary_in[field_in].items():
            if check_interval_condition(data_dictionary_in.at[idx, field_in], left_margin, right_margin, closure_type):
                if data_dictionary_out.at[idx, field_out] != mean:
                    if belong_op_out == Belong.BELONG:
                        result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {mean} but is: {data_dictionary_out.loc[idx, field_out]}")
                    elif belong_op_out == Belong.NOTBELONG:
                        result = True
                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {mean} but is: {data_dictionary_out.loc[idx, field_out]}")
            else:
                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                        pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(data_dictionary_out.at[idx, field_out])):
                    keep_no_trans_result = False
                    print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_interval_median(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                          left_margin: float, right_margin: float, closure_type: Closure,
                          belong_op_in: Belong, belong_op_out: Belong, axis_param: int = None,
                          field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the median is applied correctly on the interval
    to the data_dictionary_out respect to the data_dictionary_in

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the median
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the median
        :param left_margin: (float) left margin of the interval
        :param right_margin: (float) right margin of the interval
        :param closure_type: (Closure) closure of the interval
        :param belong_op_in: (Belong) if condition to check the invariant
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the median
        :param field_in: (str) field to apply the median
        :param field_out: (str) field to apply the median

    Returns:
        :return: True if the median is applied correctly on the interval
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
            # Calculate the median of these numeric columns
            median_value = only_numbers_df.median().median()
            # Check the data_dictionary_out positions with missing values have been replaced with the median
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                for idx, value in data_dictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type):
                            if data_dictionary_out.at[idx, col_name] != median_value:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                        else:
                            if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                    data_dictionary_out.at[idx, col_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 0:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            # Check the data_dictionary_out positions with missing values have been replaced with the median
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                median = data_dictionary_in[col_name].median()
                for idx, value in data_dictionary_in[col_name].items():
                    if np.issubdtype(type(value), np.number) or pd.isnull(value):
                        if check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type):
                            if data_dictionary_out.at[idx, col_name] != median:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                        else:
                            if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[
                                idx, col_name]) and not (
                                    pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                    data_dictionary_out.at[idx, col_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 1:
            for idx, row in data_dictionary_in.iterrows():
                numeric_data = row[row.apply(lambda x: np.isreal(x))]
                median = numeric_data.median()
                # Check if the missing values in the row have been replaced with the median in data_dictionary_out
                for col_name, value in numeric_data.items():
                    if check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type):
                        if data_dictionary_out.at[idx, col_name] != median:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(
                                data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")
        # Check the data_dictionary_out positions with missing values have been replaced with the median
        median = data_dictionary_in[field_in].median()
        for idx, value in data_dictionary_in[field_in].items():
            if check_interval_condition(data_dictionary_in.at[idx, field_in], left_margin, right_margin, closure_type):
                if data_dictionary_out.at[idx, field_out] != median:
                    if belong_op_out == Belong.BELONG:
                        result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")
                    elif belong_op_out == Belong.NOTBELONG:
                        result = True
                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")
            else:
                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                        pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(data_dictionary_out.at[idx, field_out])):
                    keep_no_trans_result = False
                    print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_interval_closest(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                           left_margin: float, right_margin: float, closure_type: Closure,
                           belong_op_in: Belong, belong_op_out: Belong, axis_param: int = None,
                           field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the closest is applied correctly on the interval
    to the data_dictionary_out respect to the data_dictionary_in

    params:
        :param data_dictionary_in: (pd.DataFrame) dataframe with the data before the closest
        :param data_dictionary_out: (pd.DataFrame) dataframe with the data after the closest
        :param left_margin: (float) left margin of the interval
        :param right_margin: (float) right margin of the interval
        :param closure_type: (Closure) closure of the interval
        :param belong_op_in: (Belong) if condition to check the invariant
        :param belong_op_out: (Belong) then condition to check the invariant
        :param axis_param: (int) axis to apply the closest
        :param field_in: (str) field to apply the closest
        :param field_out: (str) field to apply the closest

    Returns:
        :return: True if the closest is applied correctly on the interval
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param is None:
            # Select only columns with numeric data, including all numeric types (int, float, etc.)
            only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
            # Flatten the dataframe into a list of values
            flattened_values = only_numbers_df.values.flatten().tolist()
            # Create a dictionary to store the closest value for each value in the interval
            closest_values = {}
            # Iterate over the values in the interval
            for value in flattened_values:
                # Check if the value is within the interval
                if check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(value):
                    # Check if the value is already in the dictionary
                    if value not in closest_values:
                        # Find the closest value to the current value in the interval
                        closest_values[value] = find_closest_value(flattened_values, value)

            # Check if the closest values have been replaced in the data_dictionary_out
            for idx, row in data_dictionary_in.iterrows():
                for col_name in data_dictionary_in.columns:
                    if (np.isreal(row[col_name]) and check_interval_condition(row[col_name], left_margin, right_margin, closure_type)
                                    and not pd.isnull(row[col_name])):
                        if data_dictionary_out.at[idx, col_name] != closest_values[row[col_name]]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {closest_values[row[col_name]]} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {closest_values[row[col_name]]} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 0:
            for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                # Flatten the column into a list of values
                flattened_values = data_dictionary_in[col_name].values.flatten().tolist()

                # Create a dictionary to store the closest value for each value in the interval
                closest_values = {}

                # Iterate over the values in the interval
                for value in flattened_values:
                    # Check if the value is within the interval
                    if check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(value):
                        # Check if the value is already in the dictionary
                        if value not in closest_values:
                            # Find the closest value to the current value in the interval
                            closest_values[value] = find_closest_value(flattened_values, value)

                # Check if the closest values have been replaced in the data_dictionary_out
                for idx, value in data_dictionary_in[col_name].items():
                    if check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type):
                        if data_dictionary_out.at[idx, col_name] != closest_values[value]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {closest_values[value]} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {closest_values[value]} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
        elif axis_param == 1:
            for idx, row in data_dictionary_in.iterrows():
                # Flatten the row into a list of values
                flattened_values = row.values.flatten().tolist()

                # Create a dictionary to store the closest value for each value in the interval
                closest_values = {}

                # Iterate over the values in the interval
                for value in flattened_values:
                    # Check if the value is within the interval
                    if np.issubdtype(value, np.number) and check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(value):
                        # Check if the value is already in the dictionary
                        if value not in closest_values:
                            # Find the closest value to the current value in the interval
                            closest_values[value] = find_closest_value(flattened_values, value)

                # Check if the closest values have been replaced in the data_dictionary_out
                for col_name, value in row.items():
                    if np.isreal(data_dictionary_in.at[idx, col_name]) and check_interval_condition(data_dictionary_in.at[idx, col_name], left_margin, right_margin, closure_type) and not pd.isnull(data_dictionary_in.at[idx, col_name]):
                        if data_dictionary_out.at[idx, col_name] != closest_values[value]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {closest_values[value]} but is: {data_dictionary_out.loc[idx, col_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {col_name} value should be: {closest_values[value]} but is: {data_dictionary_out.loc[idx, col_name]}")
                    else:
                        if (data_dictionary_out.loc[idx, col_name] != data_dictionary_in.loc[idx, col_name]) and not (
                                pd.isnull(data_dictionary_out.at[idx, col_name]) or pd.isnull(data_dictionary_out.at[idx, col_name])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")

        # Flatten the column into a list of values
        flattened_values = data_dictionary_in[field_in].values.flatten().tolist()

        # Create a dictionary to store the closest value for each value in the interval
        closest_values = {}

        # Iterate over the values in the interval
        for value in flattened_values:
            # Check if the value is within the interval
            if check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(value):
                # Check if the value is already in the dictionary
                if value not in closest_values:
                    # Find the closest value to the current value in the interval
                    closest_values[value] = find_closest_value(flattened_values, value)

        # Check if the closest values have been replaced in the data_dictionary_out
        for idx, value in data_dictionary_in[field_in].items():
            if check_interval_condition(data_dictionary_in.at[idx, field_in], left_margin, right_margin, closure_type):
                if data_dictionary_out.at[idx, field_out] != closest_values[value]:
                    if belong_op_out == Belong.BELONG:
                        result = False
                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {closest_values[value]} but is: {data_dictionary_out.loc[idx, field_out]}")
                    elif belong_op_out == Belong.NOTBELONG:
                        result = True
                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {closest_values[value]} but is: {data_dictionary_out.loc[idx, field_out]}")
            else:
                if (data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]) and not (
                        pd.isnull(data_dictionary_out.at[idx, field_out]) or pd.isnull(data_dictionary_out.at[idx, field_out])):
                    keep_no_trans_result = False
                    print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_special_type_interpolation(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                     special_type_input: SpecialType,
                                     belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                     data_dictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                                     axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the special type interpolation is applied correctly when the input and output dataframe
    when belong_op_in is BELONG and belong_op_out is BELONG or NOTBELONG
    params:
        :param data_dictionary_in: dataframe with the data before the interpolation
        :param data_dictionary_out: dataframe with the data after the interpolation
        :param special_type_input: special type to apply the interpolation
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param data_dictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the interpolation
        :param field_in: field to apply the interpolation
        :param field_out: field to apply the interpolation

    Returns:
        :return: True if the special type interpolation is applied correctly
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    data_dictionary_in_copy = data_dictionary_in.copy()
    if field_in is None:
        if special_type_input == SpecialType.MISSING:
            if axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    data_dictionary_in_copy[col_name] = data_dictionary_in[col_name].apply(lambda x: np.nan if x in missing_values else x)

                    data_dictionary_in_copy[col_name] = data_dictionary_in_copy[col_name].interpolate(method='linear', limit_direction='both')

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in_copy[col_name] = data_dictionary_in_copy[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in_copy[col_name] = data_dictionary_in_copy[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)

                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx in data_dictionary_in.index:
                        if data_dictionary_in.at[idx, col_name] in missing_values or pd.isnull(data_dictionary_in.at[idx, col_name]):
                            if data_dictionary_out.at[idx, col_name] != data_dictionary_in_copy.at[idx, col_name]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]

                    data_dictionary_in_copy[row] = (
                        numeric_data[row].apply(lambda x: np.nan if x in missing_values else x).
                        interpolate(method='linear', limit_direction='both'))

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[row].dropna() % 1 != 0).any():
                        data_dictionary_in_copy[row] = data_dictionary_in_copy[row].round(8)
                        data_dictionary_out[row] = data_dictionary_out[row].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[row].dropna() % 1 == 0).all():
                        data_dictionary_in_copy[row] = data_dictionary_in_copy[row].round(0)
                        data_dictionary_out[row] = data_dictionary_out[row].round(0)

                for col_name in data_dictionary_in.columns:
                    for idx in data_dictionary_in.index:
                        if data_dictionary_in.at[idx, col_name] in missing_values or pd.isnull(data_dictionary_in.at[idx, col_name]):
                            if data_dictionary_out.at[idx, col_name] != data_dictionary_in_copy.at[idx, col_name]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {data_dictionary_in_copy.at[idx, col_name]} but is: {data_dictionary_out.loc[idx, col_name]}")

        elif special_type_input == SpecialType.INVALID:
            if axis_param == 0:
                for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:

                    data_dictionary_in_copy[col] = (
                        data_dictionary_in[col].apply(lambda x: np.nan if x in missing_values else x).
                        interpolate(method='linear', limit_direction='both'))

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col].dropna() % 1 != 0).any():
                        data_dictionary_in_copy[col] = data_dictionary_in_copy[col].round(8)
                        data_dictionary_out[col] = data_dictionary_out[col].round(8)
                    # Trunk the decimals to 0 if the column has no decimals and if the column hasn't nan or null values
                    elif (data_dictionary_in[col].dropna() % 1 == 0).all():
                        data_dictionary_in_copy[col] = data_dictionary_in_copy[col].round(0)
                        data_dictionary_out[col] = data_dictionary_out[col].round(0)

                # Iterate over each column
                for col in data_dictionary_in.columns:
                    # For each index in the column
                    for idx in data_dictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            data_dictionary_in_copy.at[idx, col] = data_dictionary_in.at[idx, col]

                for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx in data_dictionary_in.index:
                        if data_dictionary_in.at[idx, col] in missing_values:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in_copy.at[idx, col]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")

            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]

                    data_dictionary_in_copy[row] = (
                        numeric_data[row].apply(lambda x: np.nan if x in missing_values else x).
                        interpolate(method='linear', limit_direction='both'))

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[row].dropna() % 1 != 0).any():
                        data_dictionary_in_copy[row] = data_dictionary_in_copy[row].round(8)
                        data_dictionary_out[row] = data_dictionary_out[row].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[row].dropna() % 1 == 0).all():
                        data_dictionary_in_copy[row] = data_dictionary_in_copy[row].round(0)
                        data_dictionary_out[row] = data_dictionary_out[row].round(0)

                # Iterate over each column
                for col in data_dictionary_in.columns:
                    # For each index in the column
                    for idx in data_dictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            data_dictionary_in_copy.at[idx, col] = data_dictionary_in.at[idx, col]

                for col in data_dictionary_in.columns:
                    for idx in data_dictionary_in.index:
                        if data_dictionary_in.at[idx, col] in missing_values:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in_copy.at[idx, col]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")

        elif special_type_input == SpecialType.OUTLIER:
            if axis_param == 0:
                for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    for idx, value in data_dictionary_in[col].items():
                        if data_dictionary_outliers_mask.at[idx, col] == 1:
                            data_dictionary_in_copy.at[idx, col] = np.NaN

                    data_dictionary_in_copy[col] = data_dictionary_in_copy[col].interpolate(method='linear', limit_direction='both')

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col].dropna() % 1 != 0).any():
                        data_dictionary_in_copy[col] = data_dictionary_in_copy[col].round(8)
                        data_dictionary_out[col] = data_dictionary_out[col].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col].dropna() % 1 == 0).all():
                        data_dictionary_in_copy[col] = data_dictionary_in_copy[col].round(0)
                        data_dictionary_out[col] = data_dictionary_out[col].round(0)

                # Iterate over each column
                for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    # For each index in the column
                    for idx in data_dictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            data_dictionary_in_copy.at[idx, col] = data_dictionary_in.at[idx, col]

                for col in data_dictionary_outliers_mask.columns:
                    for idx in data_dictionary_outliers_mask.index:
                        if data_dictionary_outliers_mask.at[idx, col] == 1:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in_copy.at[idx, col]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")

            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                        if data_dictionary_outliers_mask.at[idx, col] == 1:
                            data_dictionary_in_copy.at[idx, col] = np.NaN

                    # Interpolate the row
                    data_dictionary_in_copy.loc[idx] = data_dictionary_in_copy.loc[idx].interpolate(method='linear', limit_direction='both')

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in.loc[idx].dropna() % 1 != 0).any():
                        data_dictionary_in_copy.loc[idx] = data_dictionary_in_copy.loc[idx].round(8)
                        data_dictionary_out.loc[idx] = data_dictionary_out.loc[idx].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in.loc[idx].dropna() % 1 == 0).all():
                        data_dictionary_in_copy.loc[idx] = data_dictionary_in_copy.loc[idx].round(0)
                        data_dictionary_out.loc[idx] = data_dictionary_out.loc[idx].round(0)

                # Iterate over each column
                for col in data_dictionary_in.columns:
                    # For each index in the column
                    for idx in data_dictionary_in.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_in.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            data_dictionary_in_copy.at[idx, col] = data_dictionary_in.at[idx, col]

                for col in data_dictionary_outliers_mask.columns:
                    for idx in data_dictionary_outliers_mask.index:
                        if data_dictionary_outliers_mask.at[idx, col] == 1:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in_copy.at[idx, col]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in_copy.at[idx, col]} but is: {data_dictionary_out.loc[idx, col]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("The field is not numeric")

        if special_type_input == SpecialType.MISSING:
            data_dictionary_in_copy[field_in] = (data_dictionary_in[field_in].apply(lambda x: np.nan if x in missing_values else x).
                                             interpolate(method='linear', limit_direction='both'))

            # Trunk the decimals to 8 if the column is full of floats or decimal numbers
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                data_dictionary_in_copy[field_in] = data_dictionary_in_copy[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif (data_dictionary_in[field_in].dropna() % 1 == 0).all():
                for idx in data_dictionary_in.index:
                    if data_dictionary_in_copy.at[idx, field_in] % 1 >= 0.5:
                        data_dictionary_in_copy.at[idx, field_in] = math.ceil(data_dictionary_in_copy.at[idx, field_in])
                    else:
                        data_dictionary_in_copy.at[idx, field_in] = data_dictionary_in_copy.at[idx, field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)

            for idx in data_dictionary_in.index:
                if data_dictionary_in.at[idx, field_in] in missing_values or pd.isnull(
                        data_dictionary_in.at[idx, field_in]):
                    if data_dictionary_out.at[idx, field_out] != data_dictionary_in_copy.at[idx, field_in]:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

        elif special_type_input == SpecialType.INVALID:
            data_dictionary_in_copy[field_in] = (data_dictionary_in[field_in].apply(lambda x: np.nan if x in missing_values
                else x).interpolate(method='linear', limit_direction='both'))

            # Trunk the decimals to 8 if the column is full of floats or decimal numbers
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                data_dictionary_in_copy[field_in] = data_dictionary_in_copy[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif (data_dictionary_in[field_in].dropna() % 1 == 0).all():
                for idx in data_dictionary_in.index:
                    if data_dictionary_in_copy.at[idx, field_in] % 1 >= 0.5:
                        data_dictionary_in_copy.at[idx, field_in] = math.ceil(data_dictionary_in_copy.at[idx, field_in])
                    else:
                        data_dictionary_in_copy.at[idx, field_in] = data_dictionary_in_copy.at[idx, field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)

            # For each index in the column
            for idx in data_dictionary_in.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(data_dictionary_in.at[idx, field_in]):
                    # Replace the value with the corresponding one from dataDictionary_copy_copy
                    data_dictionary_in_copy.at[idx, field_in] = data_dictionary_in.at[idx, field_in]

            for idx in data_dictionary_in_copy.index:
                if data_dictionary_in.at[idx, field_in] in missing_values:
                    if data_dictionary_out.at[idx, field_out] != data_dictionary_in_copy.at[idx, field_in]:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

        elif special_type_input == SpecialType.OUTLIER:
            for idx, value in data_dictionary_in[field_in].items():
                if data_dictionary_outliers_mask.at[idx, field_in] == 1:
                    data_dictionary_in_copy.at[idx, field_in] = np.NaN

            data_dictionary_in_copy[field_in] = data_dictionary_in_copy[field_in].interpolate(method='linear', limit_direction='both')

            # Trunk the decimals to 8 if the column is full of floats or decimal numbers
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                data_dictionary_in_copy[field_in] = data_dictionary_in_copy[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif (data_dictionary_in[field_in].dropna() % 1 == 0).all():
                for idx in data_dictionary_in.index:
                    if data_dictionary_in_copy.at[idx, field_in] % 1 >= 0.5:
                        data_dictionary_in_copy.at[idx, field_in] = math.ceil(data_dictionary_in_copy.at[idx, field_in])
                    else:
                        data_dictionary_in_copy.at[idx, field_in] = data_dictionary_in_copy.at[idx, field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)

            # For each index in the column
            for idx in data_dictionary_in.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(data_dictionary_in.at[idx, field_in]):
                    # Replace the value with the corresponding one from dataDictionary_copy_copy
                    data_dictionary_in_copy.at[idx, field_in] = data_dictionary_in.at[idx, field_in]

            for idx in data_dictionary_outliers_mask.index:
                if data_dictionary_outliers_mask.at[idx, field_in] == 1:
                    if data_dictionary_out.at[idx, field_out] != data_dictionary_in_copy.at[idx, field_in]:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in_copy.at[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    return True if result else False


def check_special_type_mean(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                            special_type_input: SpecialType,
                            belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                            data_dictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                            axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the special type mean is applied correctly when the input and output dataframes when belong_op_in and belong_op_out are BELONG
    params:
        :param data_dictionary_in: dataframe with the data before the mean
        :param data_dictionary_out: dataframe with the data after the mean
        :param special_type_input: special type to apply the mean
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param data_dictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the mean
        :param field_in: field to apply the mean
        :param field_out: field to apply the mean

    Returns:
        :return: True if the special type mean is applied correctly
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    if field_in is None:
        if special_type_input == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])

                # Calculate the mean of these numeric columns
                mean = only_numbers_df.mean().mean()
                mean_value = None

                # Check the data_dictionary_out positions with missing values have been replaced with the mean
                for col_name in data_dictionary_in.columns:

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        mean_value = round(mean, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        mean_value = round(mean, 0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if data_dictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    data_dictionary_in.at[idx, col_name]):
                                if data_dictionary_out.at[idx, col_name] != mean_value:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the data_dictionary_out positions with missing values have been replaced with the mean
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = data_dictionary_in[col_name].mean()
                    mean_value = None

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        mean_value = round(mean, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        mean_value = round(mean, 0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if data_dictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    data_dictionary_in.at[idx, col_name]):
                                if data_dictionary_out.at[idx, col_name] != mean_value:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    mean = numeric_data.mean()
                    mean_value = None

                    # Check if the missing values in the row have been replaced with the mean in data_dictionary_out
                    for col_name, value in numeric_data.items():

                        # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                        if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                            data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                            data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                            mean_value = round(mean, 8)
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                            data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                            data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                            mean_value = round(mean, 0)

                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.at[idx, col_name] != mean_value:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")

        if special_type_input == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
                # Calculate the mean of these numeric columns
                mean = only_numbers_df.mean().mean()
                mean_value = None
                # Check the data_dictionary_out positions with missing values have been replaced with the mean
                for col_name in data_dictionary_in.columns:

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        mean_value = round(mean, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        mean_value = round(mean, 0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if data_dictionary_in.at[idx, col_name] in missing_values:
                                if data_dictionary_out.at[idx, col_name] != mean_value:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the data_dictionary_out positions with missing values have been replaced with the mean
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = data_dictionary_in[col_name].mean()
                    mean_value = None

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        mean_value = round(mean, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        mean_value = round(mean, 0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if data_dictionary_in.at[idx, col_name] in missing_values:
                                if data_dictionary_out.at[idx, col_name] != mean:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    mean = numeric_data.mean()
                    mean_value = None
                    # Check if the missing values in the row have been replaced with the mean in data_dictionary_out
                    for col_name, value in numeric_data.items():

                        # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                        if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                            data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                            data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                            mean_value = round(mean, 8)
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                            data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                            data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                            mean_value = round(mean, 0)

                        if value in missing_values:
                            if data_dictionary_out.at[idx, col_name] != mean_value:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")

        if special_type_input == SpecialType.OUTLIER:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
                # Calculate the mean of these numeric columns
                mean = only_numbers_df.mean().mean()
                mean_value = None
                # Replace the missing values with the mean of the entire DataFrame using lambda
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:

                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        mean_value = round(mean, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        mean_value = round(mean, 0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if data_dictionary_outliers_mask.at[idx, col_name] == 1:
                            if data_dictionary_out.at[idx, col_name] != mean_value:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")

            if axis_param == 0:  # Iterate over each column
                for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    mean = data_dictionary_in[col].mean()
                    mean_value = None
                    # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                    if (data_dictionary_in[col].dropna() % 1 != 0).any():
                        data_dictionary_in[col] = data_dictionary_in[col].round(8)
                        data_dictionary_out[col] = data_dictionary_out[col].round(8)
                        mean_value = round(mean, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col].dropna() % 1 == 0).all():
                        data_dictionary_in[col] = data_dictionary_in[col].round(0)
                        data_dictionary_out[col] = data_dictionary_out[col].round(0)
                        mean_value = round(mean, 0)

                    for idx, value in data_dictionary_in[col].items():
                        if data_dictionary_outliers_mask.at[idx, col] == 1:
                            if data_dictionary_out.at[idx, col] != mean_value:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col]}")

            elif axis_param == 1:  # Iterate over each row
                for idx, row in data_dictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    mean = numeric_data.mean()
                    mean_value = None

                    for col_name, value in numeric_data.items():

                        # Trunk the decimals to 8 if the column is full of floats or decimal numbers
                        if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                            data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                            data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                            mean_value = round(mean, 8)
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                            data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                            data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                            mean_value = round(mean, 0)

                        if data_dictionary_outliers_mask.at[idx, col_name] == 1:
                            if data_dictionary_out.at[idx, col_name] != mean_value:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")

        if special_type_input == SpecialType.MISSING:
            # Check the data_dictionary_out positions with missing values have been replaced with the mean
            mean = data_dictionary_in[field_in].mean()
            mean_value = None
            # Trunk the decimals to 8 if the column is full of floats or decimal numbers
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
                mean_value = round(mean, 8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif (data_dictionary_in[field_in].dropna() % 1 == 0).all():
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)
                mean_value = round(mean, 0)

            for idx, value in data_dictionary_in[field_in].items():
                if value in missing_values or pd.isnull(value):
                    if data_dictionary_out.at[idx, field_out] != mean_value:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, field_out]}")

        if special_type_input == SpecialType.INVALID:
            # Check the data_dictionary_out positions with missing values have been replaced with the mean
            mean = data_dictionary_in[field_in].mean()
            mean_value = None

            # Trunk the decimals to 8 if the column is full of floats or decimal numbers
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
                mean_value = round(mean, 8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif (data_dictionary_in[field_in].dropna() % 1 == 0).all():
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)
                mean_value = round(mean, 0)

            for idx, value in data_dictionary_in[field_in].items():
                if value in missing_values:
                    if data_dictionary_out.at[idx, field_out] != mean_value:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, field_out]}")


        if special_type_input == SpecialType.OUTLIER:

            mean = data_dictionary_in[field_in].mean()
            mean_value = None
            # Trunk the decimals to 8 if the column is full of floats or decimal numbers
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
                mean_value = round(mean, 8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif (data_dictionary_in[field_in].dropna() % 1 == 0).all():
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)
                mean_value = round(mean, 0)

            for idx, value in data_dictionary_in[field_in].items():
                if data_dictionary_outliers_mask.at[idx, field_in] == 1:

                    if data_dictionary_out.at[idx, field_out] != mean_value:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {mean_value} but is: {data_dictionary_out.loc[idx, field_out]}")


    return True if result else False


def check_special_type_median(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                              special_type_input: SpecialType,
                              belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                              data_dictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                              axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the special type median is applied correctly when the input and output dataframes when belong_op_in and belong_op_out are BELONG
    params:
        :param data_dictionary_in: dataframe with the data before the median
        :param data_dictionary_out: dataframe with the data after the median
        :param special_type_input: special type to apply the median
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param data_dictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the median
        :param field_in: field to apply the median
        :param field_out: field to apply the median

    Returns:
        :return: True if the special type median is applied correctly
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    if field_in is None:
        if special_type_input == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median = only_numbers_df.median().median()
                median_value = None
                # Check the data_dictionary_out positions with missing values have been replaced with the median
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:

                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        median_value = round(median, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        median_value = round(median, 0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if pd.isnull(value):
                            if data_dictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    data_dictionary_in.at[idx, col_name]):
                                if data_dictionary_out.at[idx, col_name] != median_value:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the data_dictionary_out positions with missing values have been replaced with the median
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    median = data_dictionary_in[col_name].median()

                    # Trunk the decimals to 8 if the column is float or if it has decimals
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        median = round(median, 8)
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif data_dictionary_in[col_name].apply(lambda x: x % 1 == 0).all():
                        median = round(median, 0)
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number) or pd.isnull(value):
                            if data_dictionary_in.at[idx, col_name] in missing_values or pd.isnull(
                                    data_dictionary_in.at[idx, col_name]):
                                if data_dictionary_out.at[idx, col_name] != median:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    median = numeric_data.median()

                    # Trunk the decimals to 8 if the column is float or if it has decimals
                    if (data_dictionary_in[row].dropna() % 1 != 0).any():
                        median = round(median, 8)
                        data_dictionary_in[row] = data_dictionary_in[row].round(8)
                        data_dictionary_out[row] = data_dictionary_out[row].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif data_dictionary_in[row].apply(lambda x: x % 1 == 0).all():
                        median = round(median, 0)
                        data_dictionary_in[row] = data_dictionary_in[row].round(0)
                        data_dictionary_out[row] = data_dictionary_out[row].round(0)

                    # Check if the missing values in the row have been replaced with the median in data_dictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.at[idx, col_name] != median:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")

        if special_type_input == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median = only_numbers_df.median().median()
                median_value = None
                # Check the data_dictionary_out positions with missing values have been replaced with the median
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:

                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        median_value = round(median, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        median_value = round(median, 0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if data_dictionary_in.at[idx, col_name] in missing_values:
                            if data_dictionary_out.at[idx, col_name] != median_value:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 0:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                # Check the data_dictionary_out positions with missing values have been replaced with the median
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    median = data_dictionary_in[col_name].median()

                    # Trunk the decimals to 8 if the column is float or if it has decimals
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        median = round(median, 8)
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif  data_dictionary_in[col_name].apply(lambda x: x % 1 == 0).all():
                        median = round(median, 0)
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if np.issubdtype(type(value), np.number):
                            if data_dictionary_in.at[idx, col_name] in missing_values:
                                if data_dictionary_out.at[idx, col_name] != median:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")

            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    median = numeric_data.median()

                    # Trunk the decimals to 8 if the column is float or if it has decimals
                    if (data_dictionary_in[row].dropna() % 1 != 0).any():
                        median = round(median, 8)
                        data_dictionary_in[row] = data_dictionary_in[row].round(8)
                        data_dictionary_out[row] = data_dictionary_out[row].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif data_dictionary_in[row].dtype == int or data_dictionary_in[row].apply(lambda x: x % 1 == 0).all():
                        median = round(median, 0)
                        data_dictionary_in[row] = data_dictionary_in[row].round(0)
                        data_dictionary_out[row] = data_dictionary_out[row].round(0)

                    # Check if the missing values in the row have been replaced with the median in data_dictionary_out
                    for col_name, value in numeric_data.items():
                        if value in missing_values:
                            if data_dictionary_out.at[idx, col_name] != median:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")

        if special_type_input == SpecialType.OUTLIER:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_in.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median = only_numbers_df.median().median()
                median_value = None
                # Replace the missing values with the median of the entire DataFrame using lambda
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        median_value = round(median, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        median_value = round(median, 0)

                    for idx, value in data_dictionary_in[col_name].items():
                        if data_dictionary_outliers_mask.at[idx, col_name] == 1:
                            if data_dictionary_out.at[idx, col_name] != median_value:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median_value} but is: {data_dictionary_out.loc[idx, col_name]}")

            if axis_param == 0:  # Iterate over each column
                for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    median = data_dictionary_in[col].median()

                    # Trunk the decimals to 8 if the column is float or if it has decimals
                    if (data_dictionary_in[col].dropna() % 1 != 0).any():
                        median = round(median, 8)
                        data_dictionary_in[col] = data_dictionary_in[col].round(8)
                        data_dictionary_out[col] = data_dictionary_out[col].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif data_dictionary_in[col].apply(lambda x: x % 1 == 0).all():
                        median = round(median, 0)
                        data_dictionary_in[col] = data_dictionary_in[col].round(0)
                        data_dictionary_out[col] = data_dictionary_out[col].round(0)

                    for idx, value in data_dictionary_in[col].items():
                        if data_dictionary_outliers_mask.at[idx, col] == 1:
                            if data_dictionary_out.at[idx, col] != median:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {median} but is: {data_dictionary_out.loc[idx, col]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {median} but is: {data_dictionary_out.loc[idx, col]}")

            elif axis_param == 1:  # Iterate over each row
                for idx, row in data_dictionary_in.iterrows():
                    numeric_data = row[row.apply(lambda x: np.isreal(x))]
                    median = numeric_data.median()

                    # Trunk the decimals to 8 if the column is float or if it has decimals
                    if (data_dictionary_in[row].dropna() % 1 != 0).any():
                        median = round(median, 8)
                        data_dictionary_in[row] = data_dictionary_in[row].round(8)
                        data_dictionary_out[row] = data_dictionary_out[row].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif data_dictionary_in[row].apply(lambda x: x % 1 == 0).all():
                        median = round(median, 0)
                        data_dictionary_in[row] = data_dictionary_in[row].round(0)
                        data_dictionary_out[row] = data_dictionary_out[row].round(0)

                    for col_name, value in numeric_data.items():
                        if data_dictionary_outliers_mask.at[idx, col_name] == 1:
                            if data_dictionary_out.at[idx, col_name] != median:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col_name} value should be: {median} but is: {data_dictionary_out.loc[idx, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")

        if special_type_input == SpecialType.MISSING:
            # Check the data_dictionary_out positions with missing values have been replaced with the median
            median = data_dictionary_in[field_in].median()

            # Trunk the decimals to 8 if the column is float or if it has decimals
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                median = round(median, 8)
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif data_dictionary_in[field_in].apply(lambda x: x % 1 == 0).all():
                median = round(median, 0)
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)

            for idx, value in data_dictionary_in[field_in].items():
                if value in missing_values or pd.isnull(value):
                    if data_dictionary_out.at[idx, field_out] != median:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")

        if special_type_input == SpecialType.INVALID:
            # Check the data_dictionary_out positions with missing values have been replaced with the median
            median = data_dictionary_in[field_in].median()

            # Trunk the decimals to 8 if the column is float or if it has decimals
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                median = round(median, 8)
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif data_dictionary_in[field_in].apply(lambda x: x % 1 == 0).all():
                median = round(median, 0)
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)

            for idx, value in data_dictionary_in[field_in].items():
                if value in missing_values:
                    if data_dictionary_out.at[idx, field_out] != median:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")

        if special_type_input == SpecialType.OUTLIER:

            median = data_dictionary_in[field_in].median()

            # Trunk the decimals to 8 if the column is float or if it has decimals
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                median = round(median, 8)
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif data_dictionary_in[field_in].apply(lambda x: x % 1 == 0).all():
                median = round(median, 0)
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)

            for idx, value in data_dictionary_in[field_in].items():
                if data_dictionary_outliers_mask.at[idx, field_in] == 1:

                    if data_dictionary_out.at[idx, field_out] != median:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                            print_and_log(f"Row: {idx} and column: {field_out} value should be: {median} but is: {data_dictionary_out.loc[idx, field_out]}")

    return True if result else False


def check_special_type_closest(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                               special_type_input: SpecialType, belong_op_in: Belong, belong_op_out: Belong,
                               data_dictionary_outliers_mask: pd.DataFrame = None, missing_values: list = None,
                               axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the special type closest is applied correctly when the input and output dataframes when belong_op_in is Belong
    params:
        :param data_dictionary_in: dataframe with the data before the closest
        :param data_dictionary_out: dataframe with the data after the closest
        :param special_type_input: special type to apply the closest
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param data_dictionary_outliers_mask: dataframe with the mask of the outliers
        :param missing_values: list of missing values
        :param axis_param: axis to apply the closest
        :param field_in: field to apply the closest
        :param field_out: field to apply the closest

    Returns:
        :return: True if the special type closest is applied correctly
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    if field_in is None:
        if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
            if axis_param is None:
                only_numbers_df=data_dictionary_in.select_dtypes(include=[np.number])
                # Flatten the DataFrame into a single series of values
                flattened_values = only_numbers_df.values.flatten().tolist()
                # Create a dictionary to store the closest value for each missing value
                closest_values = {}
                # For each missing value, find the closest numeric value in the flattened series
                for missing_value in missing_values:
                    if missing_value not in closest_values:
                        closest_values[missing_value] = find_closest_value(flattened_values, missing_value)
                # Replace the missing values with the closest numeric values
                for i in range(len(data_dictionary_in.index)):
                    for j in range(len(data_dictionary_in.columns)):
                        current_value = data_dictionary_in.iloc[i, j]
                        if current_value in closest_values:
                            if data_dictionary_out.iloc[i, j] != closest_values[current_value]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {i} and column: {j} value should be: {closest_values[current_value]} but is: {data_dictionary_out.loc[i, j]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {i} and column: {j} value should be: {closest_values[current_value]} but is: {data_dictionary_out.loc[i, j]}")

            elif axis_param == 0:
                # Iterate over each column
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    # Get the missing values in the current column
                    missing_values_in_col = [val for val in missing_values if val in data_dictionary_in[col_name].values]

                    # If there are no missing values in the column, skip the rest of the loop
                    if not missing_values_in_col:
                        continue

                    # Flatten the column into a list of values
                    flattened_values = data_dictionary_in[col_name].values.flatten().tolist()

                    # Create a dictionary to store the closest value for each missing value
                    closest_values = {}

                    # For each missing value IN the column (more efficient), find the closest numeric value in the flattened series
                    for missing_value in missing_values_in_col:
                        if missing_value not in closest_values:
                            closest_values[missing_value] = find_closest_value(flattened_values, missing_value)

                    # Replace the missing values with the closest numeric values in the column
                    for i in range(len(data_dictionary_in.index)):
                        current_value = data_dictionary_in.at[i, col_name]
                        if current_value in closest_values:
                            if data_dictionary_out.at[i, col_name] != closest_values[current_value]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {i} and column: {col_name} value should be: {closest_values[current_value]} but is: {data_dictionary_out.loc[i, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {i} and column: {col_name} value should be: {closest_values[current_value]} but is: {data_dictionary_out.loc[i, col_name]}")

            elif axis_param == 1:
                # Iterate over each row
                for row_idx in range(len(data_dictionary_in.index)):
                    # Get the numeric values in the current row
                    numeric_values_in_row = data_dictionary_in.iloc[row_idx].select_dtypes(include=[np.number]).values.tolist()

                    # Get the missing values in the current row
                    missing_values_in_row = [val for val in missing_values if val in numeric_values_in_row]

                    # If there are no missing values in the row, skip the rest of the loop
                    if not missing_values_in_row and not pd.isnull(data_dictionary_in.iloc[row_idx]).any():
                        continue

                    # Create a dictionary to store the closest value for each missing value
                    closest_values = {}

                    # For each missing value IN the row (more efficient), find the closest numeric value in the numeric values
                    for missing_value in missing_values_in_row:
                        if missing_value not in closest_values:
                            closest_values[missing_value] = find_closest_value(numeric_values_in_row, missing_value)

                    # Replace the missing values with the closest numeric values in the row
                    for col_name in data_dictionary_in.columns:
                        current_value = data_dictionary_in.at[row_idx, col_name]
                        if current_value in closest_values:
                            if data_dictionary_out.at[row_idx, col_name] != closest_values[current_value]:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_idx} and column: {col_name} value should be: {closest_values[current_value]} but is: {data_dictionary_out.loc[row_idx, col_name]}")
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_idx} and column: {col_name} value should be: {closest_values[current_value]} but is: {data_dictionary_out.loc[row_idx, col_name]}")

        if special_type_input == SpecialType.OUTLIER:
            if axis_param is None:
                minimum_valid, maximum_valid = outlier_closest(data_dictionary=data_dictionary_in,
                                                               axis_param=None, field=None)
                minimum_valid_rounded = None
                maximum_valid_rounded = None
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                        minimum_valid_rounded = round(minimum_valid, 8)
                        maximum_valid_rounded = round(maximum_valid, 8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif (data_dictionary_in[col_name].dropna() % 1 == 0).all():
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)
                        minimum_valid_rounded = round(minimum_valid, 0)
                        maximum_valid_rounded = round(maximum_valid, 0)

                    for i in range(len(data_dictionary_in.index)):
                        if data_dictionary_outliers_mask.at[i, col_name] == 1:
                            if data_dictionary_in.at[i, col_name] > maximum_valid:
                                if data_dictionary_out.at[i, col_name] != maximum_valid:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {i} and column: {col_name} value should be: {maximum_valid_rounded} but is: {data_dictionary_out.loc[i, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {i} and column: {col_name} value should be: {maximum_valid_rounded} but is: {data_dictionary_out.loc[i, col_name]}")
                            elif data_dictionary_in.at[i, col_name] < minimum_valid:
                                if data_dictionary_out.at[i, col_name] != minimum_valid:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {i} and column: {col_name} value should be: {minimum_valid_rounded} but is: {data_dictionary_out.loc[i, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {i} and column: {col_name} value should be: {minimum_valid_rounded} but is: {data_dictionary_out.loc[i, col_name]}")

            elif axis_param == 0:
                # Checks the outlier values in the input with the closest numeric values in the output
                for col_name in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                    minimum_valid, maximum_valid = outlier_closest(data_dictionary=data_dictionary_in,
                                                                   axis_param=0, field=col_name)

                    # Trunk the decimals to 8 if the column is float or if it has decimals
                    if (data_dictionary_in[col_name].dropna() % 1 != 0).any():
                        minimum_valid = round(minimum_valid, 8)
                        maximum_valid = round(maximum_valid, 8)
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(8)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(8)
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    elif data_dictionary_in[col_name].apply(lambda x: x % 1 == 0).all():
                        minimum_valid = round(minimum_valid, 0)
                        maximum_valid = round(maximum_valid, 0)
                        data_dictionary_in[col_name] = data_dictionary_in[col_name].round(0)
                        data_dictionary_out[col_name] = data_dictionary_out[col_name].round(0)

                    for i in range(len(data_dictionary_in.index)):
                        if data_dictionary_outliers_mask.at[i, col_name] == 1:
                            if data_dictionary_in.at[i, col_name] > maximum_valid:
                                if data_dictionary_out.at[i, col_name] != maximum_valid:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {i} and column: {col_name} value should be: {maximum_valid} but is: {data_dictionary_out.loc[i, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {i} and column: {col_name} value should be: {maximum_valid} but is: {data_dictionary_out.loc[i, col_name]}")
                            elif data_dictionary_in.at[i, col_name] < minimum_valid:
                                if data_dictionary_out.at[i, col_name] != minimum_valid:
                                    if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {i} and column: {col_name} value should be: {minimum_valid} but is: {data_dictionary_out.loc[i, col_name]}")
                                    elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {i} and column: {col_name} value should be: {minimum_valid} but is: {data_dictionary_out.loc[i, col_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")

        if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
            # Get the missing values in the current column
            missing_values_in_col = [val for val in missing_values if val in data_dictionary_in[field_in].values]
            # If there are no missing values in the column, skip the rest of the loop
            if missing_values_in_col or pd.isnull(data_dictionary_in[field_in]).any():
                # Flatten the column into a list of values
                flattened_values = data_dictionary_in[field_in].values.flatten().tolist()
                # Create a dictionary to store the closest value for each missing value
                closest_values = {}
                # For each missing value IN the column (more efficient), find the closest numeric value in the flattened series
                for missing_value in missing_values_in_col:
                    if missing_value not in closest_values:
                        closest_values[missing_value] = find_closest_value(flattened_values, missing_value)
                # Replace the missing values with the closest numeric values in the column
                for i in range(len(data_dictionary_in.index)):
                    current_value = data_dictionary_in.at[i, field_in]
                    if current_value in closest_values:
                        if data_dictionary_out.at[i, field_out] != closest_values[current_value]:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {i} and column: {field_out} value should be: {closest_values[current_value]} but is: {data_dictionary_out.loc[i, field_out]}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {i} and column: {field_out} value should be: {closest_values[current_value]} but is: {data_dictionary_out.loc[i, field_out]}")

                    elif pd.isnull(current_value) and special_type_input == SpecialType.MISSING:
                        raise ValueError("Missing value not found in the closest_values dictionary")

        if special_type_input == SpecialType.OUTLIER:
            minimum_valid, maximum_valid = outlier_closest(data_dictionary=data_dictionary_in,
                                                           axis_param=None, field=field_in)
            #TODO: Check if the rounding is correct
            # Trunk the decimals to 8 if the column is float or if it has decimals
            if (data_dictionary_in[field_in].dropna() % 1 != 0).any():
                minimum_valid = round(minimum_valid, 8)
                maximum_valid = round(maximum_valid, 8)
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(8)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(8)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            elif (data_dictionary_in[field_in].dropna() % 1 == 0).all():
                minimum_valid = round(minimum_valid, 0)
                maximum_valid = round(maximum_valid, 0)
                data_dictionary_in[field_in] = data_dictionary_in[field_in].round(0)
                data_dictionary_out[field_out] = data_dictionary_out[field_out].round(0)

            # Checks the outlier values in the input with the closest numeric values in the output
            for i in range(len(data_dictionary_in.index)):
                if data_dictionary_outliers_mask.at[i, field_in] == 1:
                    if data_dictionary_in.at[i, field_in] > maximum_valid:
                        if data_dictionary_out.at[i, field_out] != maximum_valid:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {i} and column: {field_out} value should be: {maximum_valid} but is: {data_dictionary_out.loc[i, field_out]}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {i} and column: {field_out} value should be: {maximum_valid} but is: {data_dictionary_out.loc[i, field_out]}")
                    elif data_dictionary_in.at[i, field_in] < minimum_valid:
                        if data_dictionary_out.at[i, field_out] != minimum_valid:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {i} and column: {field_out} value should be: {minimum_valid} but is: {data_dictionary_out.loc[i, field_out]}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {i} and column: {field_out} value should be: {minimum_valid} but is: {data_dictionary_out.loc[i, field_out]}")

    return True if result else False

def check_special_type_most_frequent(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                     special_type_input: SpecialType, belong_op_out: Belong, missing_values: list = None,
                                     axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the special type most frequent is applied correctly when the input and output dataframes when belong_op_in is Belong and belong_op_out is BELONG

    params:

    :param data_dictionary_in: (pd.DataFrame) Dataframe with the data before the most frequent
    :param data_dictionary_out: (pd.DataFrame) Dataframe with the data after the most frequent
    :param special_type_input: (SpecialType) Special type to apply the most frequent
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the most frequent
    :param field_in: (str) Field to apply the most frequent
    :param field_out: (str) Field to apply the most frequent
    :param belong_op_out: (Belong) Then condition to check the invariant

    :return: True if the special type most frequent is applied correctly, False otherwise
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if axis_param == 0:
            if special_type_input == SpecialType.MISSING:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    most_frequent = data_dictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in data_dictionary_in[column_name].items():
                        if pd.isnull(value):
                            if data_dictionary_out.loc[row_index, column_name] != most_frequent:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_name} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        elif missing_values is None:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name]:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:
                            if value in missing_values:
                                if data_dictionary_out.loc[row_index, column_name] != most_frequent:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_name} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            else:
                                if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                    row_index, column_name]:
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")

            else:  # It works for invalid values and outliers
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    most_frequent = data_dictionary_in[column_name].value_counts().idxmax()
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, column_name] != most_frequent:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_name} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:  # If the output value isn't a missing value and isn't equal to the
                            # original value, then return False
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name] and not (
                                    pd.isnull(data_dictionary_in.loc[row_index, column_name]) and pd.isnull(
                                    data_dictionary_out.loc[row_index, column_name])) and not (special_type_input == SpecialType.MISSING):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
        elif axis_param == 1:
            if special_type_input == SpecialType.MISSING:
                # Instead of iterating over the columns, we iterate over the rows to check the derived type
                for row_index, row in data_dictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if pd.isnull(value):
                            if data_dictionary_out.loc[row_index, column_index] != most_frequent:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                        elif missing_values is None:
                            if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                row_index, column_index]:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
                        else:
                            if value in missing_values:
                                if data_dictionary_out.loc[row_index, column_index] != most_frequent:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                            else:
                                if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                    row_index, column_index]:
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
            else:  # It works for invalid values and outliers
                for row_index, row in data_dictionary_in.iterrows():
                    most_frequent = row.value_counts().idxmax()
                    for column_index, value in row.items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, column_index] != most_frequent:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(data_dictionary_in.loc[row_index, column_index]) and pd.isnull(
                                    data_dictionary_out.loc[row_index, column_index])) and not (special_type_input == SpecialType.MISSING):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
        elif axis_param is None:
            most_frequent = data_dictionary_in.stack().value_counts().idxmax()
            if special_type_input == SpecialType.MISSING:
                for row_index, row in data_dictionary_in.iterrows():
                    for column_index, value in row.items():
                        if pd.isnull(value):
                            if data_dictionary_out.loc[row_index, column_index] != most_frequent:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                        elif missing_values is None:
                            if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                row_index, column_index]:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
                        else:
                            if value in missing_values:
                                if data_dictionary_out.loc[row_index, column_index] != most_frequent:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                            else:
                                if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                    row_index, column_index]:
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
            else:  # It works for invalid values and outliers
                for row_index, row in data_dictionary_in.iterrows():
                    for column_index, value in row.items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, column_index] != most_frequent:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {row_index} and column: {column_index} value should be: {most_frequent} but is: {data_dictionary_out.loc[row_index, column_index]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(data_dictionary_in.loc[row_index, column_index]) and pd.isnull(
                                    data_dictionary_out.loc[row_index, column_index])) and not (special_type_input == SpecialType.MISSING):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field is not in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            most_frequent = data_dictionary_in[field_in].value_counts().idxmax()
            if special_type_input == SpecialType.MISSING:
                for idx, value in data_dictionary_in[field_in].items():
                    if pd.isnull(value):
                        if data_dictionary_out.loc[idx, field_out] != most_frequent:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {most_frequent} but is: {data_dictionary_out.loc[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {most_frequent} but is: {data_dictionary_out.loc[idx, field_out]}")
                    elif missing_values is None:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                    else:
                        if value in missing_values:
                            if data_dictionary_out.loc[idx, field_out] != most_frequent:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {most_frequent} but is: {data_dictionary_out.loc[idx, field_out]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {field_out} value should be: {most_frequent} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:
                            if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
            else:  # It works for invalid values and outliers
                for idx, value in data_dictionary_in[field_in].items():
                    if value in missing_values:
                        if data_dictionary_out.loc[idx, field_out] != most_frequent:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {most_frequent} but is: {data_dictionary_out.loc[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {most_frequent} but is: {data_dictionary_out.loc[idx, field_out]}")
                    else:
                        if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                                pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(
                            data_dictionary_out.loc[idx, field_out])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_special_type_previous(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                special_type_input: SpecialType, belong_op_out: Belong = Belong.BELONG,
                                missing_values: list = None, axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the derived type previous value is applied correctly when the input and output dataframes when belong_op_in and belong_op_out are both BELONG

    params:
    :param data_dictionary_in: (pd.DataFrame) Dataframe with the data before the previous
    :param data_dictionary_out: (pd.DataFrame) Dataframe with the data after the previous
    :param special_type_input: (SpecialType) Special type to apply the previous
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the previous
    :param field_in: (str) Field to apply the previous
    :param field_out: (str) Field to apply the previous
    :param belong_op_out: (Belong) Then condition to check the invariant

    :return: True if the derived type previous value is applied correctly, False otherwise
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        # Check the previous value of the missing values
        if axis_param == 0:
            if special_type_input == SpecialType.MISSING:
                # Manual check of the previous operacion to the missing values in the columns
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if row_index == 0:
                                if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                    row_index, column_name] and not (
                                        pd.isnull(data_dictionary_in.loc[row_index, column_name]) and pd.isnull(
                                    data_dictionary_out.loc[row_index, column_name])):
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            else:
                                if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                    row_index - 1, column_name] and (
                                        not pd.isnull(data_dictionary_in.loc[row_index, column_name]) and pd.isnull(
                                    data_dictionary_out.loc[row_index - 1, column_name])):
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index - 1, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index - 1, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name] and not (
                                    pd.isnull(data_dictionary_in.loc[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.loc[row_index, column_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values:
                            if row_index == 0:
                                if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                    row_index, column_name]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            else:
                                # Check that is posible to access to the previous row
                                if row_index - 1 in data_dictionary_in.index:
                                    if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                        row_index - 1, column_name]:
                                        if belong_op_out == Belong.BELONG:
                                            result = False
                                            print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index - 1, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                        elif belong_op_out == Belong.NOTBELONG:
                                            result = True
                                            print_and_log(f"Row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index - 1, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name] and not (pd.isnull(
                                data_dictionary_in.loc[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.loc[row_index, column_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")

        elif axis_param == 1:
            if special_type_input == SpecialType.MISSING:
                for row_index, row in data_dictionary_in.iterrows():
                    for column_index in range(len(data_dictionary_in.columns)):
                        value = data_dictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            if column_index == 0:
                                if data_dictionary_out.at[row_index, column_index] != data_dictionary_in.at[
                                    row_index, column_index]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_index]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_index]}")
                            else:
                                if column_index - 1 in data_dictionary_in.columns:
                                    if data_dictionary_out.at[row_index, column_index] != data_dictionary_in.at[
                                        row_index, column_index - 1]:
                                        if belong_op_out == Belong.BELONG:
                                            result = False
                                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index - 1]} but is: {data_dictionary_out.at[row_index, column_index]}")
                                        elif belong_op_out == Belong.NOTBELONG:
                                            result = True
                                            print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index - 1]} but is: {data_dictionary_out.at[row_index, column_index]}")
                        else:
                            if data_dictionary_out.at[row_index, column_index] != data_dictionary_in.at[
                                row_index, column_index]:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_index]}")

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in data_dictionary_in.iterrows():
                    for column_index in range(len(data_dictionary_in.columns)):
                        value = data_dictionary_in.iloc[row_index, column_index]
                        if value in missing_values:
                            if column_index == 0:
                                if data_dictionary_out.iloc[row_index, column_index] != data_dictionary_in.iloc[
                                    row_index, column_index]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.iloc[row_index, column_index]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.iloc[row_index, column_index]}")
                            else:
                                if column_index - 1 in data_dictionary_in.columns:
                                    if data_dictionary_out.iloc[row_index, column_index] != data_dictionary_in.iloc[
                                        row_index, column_index - 1]:
                                        if belong_op_out == Belong.BELONG:
                                            result = False
                                            print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index - 1]} but is: {data_dictionary_out.iloc[row_index, column_index]}")
                                        elif belong_op_out == Belong.NOTBELONG:
                                            result = True
                                            print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index - 1]} but is: {data_dictionary_out.iloc[row_index, column_index]}")
                        else:
                            if data_dictionary_out.iloc[row_index, column_index] != data_dictionary_in.iloc[
                                row_index, column_index] and not (
                                    pd.isnull(data_dictionary_in.iloc[row_index, column_index]) and pd.isnull(
                                    data_dictionary_out.iloc[row_index, column_index])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.iloc[row_index, column_index]} but is: {data_dictionary_out.iloc[row_index, column_index]}")

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field is not in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            if special_type_input == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in data_dictionary_in[field_in].items():
                        if value in missing_values or pd.isnull(value):
                            if idx == 0:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                            else:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx - 1, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx - 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx - 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:
                            if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                else: # missing_values is None
                    for idx, value in data_dictionary_in[field_in].items():
                        if pd.isnull(value):
                            if idx == 0:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                            else:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx - 1, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx - 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx - 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:
                            if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

            elif special_type_input == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in data_dictionary_in[field_in].items():
                        if value in missing_values:
                            if idx == 0:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                            else:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx - 1, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx - 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx - 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:
                            if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(
                                data_dictionary_out.loc[idx, field_out])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_special_type_next(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                            special_type_input: SpecialType, belong_op_out: Belong = Belong.BELONG,
                            missing_values: list = None, axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check if the derived type next value is applied correctly when the input and output
    dataframes when belong_op_in and belong_op_out are BELONG
    :param data_dictionary_in: (pd.DataFrame) Dataframe with the data before the next
    :param data_dictionary_out: (pd.DataFrame) Dataframe with the data after the next
    :param special_type_input: (SpecialType) Special type to apply the next
    :param missing_values: (list) List of missing values
    :param axis_param: (int) Axis to apply the next
    :param field_in: (str) Field to apply the next
    :param field_out: (str) Field to apply the next
    :param belong_op_out: (Belong) Belong condition to check the invariant

    :return: True if the derived type next value is applied correctly, False otherwise
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        # Define the lambda function to replace the values within missing values by the value of the next position
        if axis_param == 0:
            if special_type_input == SpecialType.MISSING:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if (value in missing_values or pd.isnull(value)) and row_index < len(data_dictionary_in) - 1:
                            if row_index == len(data_dictionary_in) - 1:
                                if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                    row_index, column_name] and not (
                                        pd.isnull(data_dictionary_in.loc[row_index, column_name]) and pd.isnull(
                                    data_dictionary_out.loc[row_index, column_name])):
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            else:
                                if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                    row_index + 1, column_name] and not (
                                         pd.isnull(data_dictionary_out.loc[row_index, column_name]) and pd.isnull(data_dictionary_in.loc[row_index + 1, column_name])):
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index + 1, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index + 1, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name] and not (
                                    pd.isnull(data_dictionary_in.loc[row_index, column_name]) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values:
                            if row_index == len(data_dictionary_in) - 1:
                                if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                    row_index, column_name]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            else:
                                if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                    row_index + 1, column_name]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index + 1, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index + 1, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_name] != data_dictionary_in.loc[
                                row_index, column_name] and not (pd.isnull(
                                data_dictionary_in.loc[row_index, column_name]) and pd.isnull(
                                data_dictionary_out.loc[row_index, column_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")

        elif axis_param == 1:
            if special_type_input == SpecialType.MISSING:
                for row_index, row in data_dictionary_in.iterrows():
                    for column_index in range(len(data_dictionary_in.columns)):
                        value = data_dictionary_in.at[row_index, column_index]
                        if value in missing_values or pd.isnull(value):
                            if column_index == len(row) - 1:
                                if data_dictionary_out.at[row_index, column_index] != data_dictionary_in.at[
                                    row_index, column_index]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_index]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_index]}")
                            else:
                                if data_dictionary_out.at[row_index, column_index] != data_dictionary_in.at[
                                    row_index, column_index + 1]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index + 1]} but is: {data_dictionary_out.at[row_index, column_index]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index + 1]} but is: {data_dictionary_out.at[row_index, column_index]}")
                        else:
                            if data_dictionary_out.at[row_index, column_index] != data_dictionary_in.at[
                                row_index, column_index] and not (
                                    pd.isnull(data_dictionary_in.at[row_index, column_index]) and pd.isnull(
                                data_dictionary_out.at[row_index, column_index])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.at[row_index, column_index]} but is: {data_dictionary_out.at[row_index, column_index]}")

            else:  # SPECIAL_TYPE is INVALID or OUTLIER
                for row_index, row in data_dictionary_in.iterrows():
                    for column_index, value in row.items():
                        column_names = data_dictionary_in.columns.tolist()
                        current_column_index = column_names.index(column_index)
                        if current_column_index < len(column_names) - 1:  # check if it's not the last column
                            next_column_name = column_names[current_column_index + 1]
                        if value in missing_values:
                            if column_index == len(row) - 1:
                                if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                    row_index, column_index]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")
                            else:
                                if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                    row_index, next_column_name]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, next_column_name]} but is: {data_dictionary_out.loc[row_index, column_index]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, next_column_name]} but is: {data_dictionary_out.loc[row_index, column_index]}")
                        else:
                            if data_dictionary_out.loc[row_index, column_index] != data_dictionary_in.loc[
                                row_index, column_index] and not (
                                    pd.isnull(data_dictionary_in.loc[row_index, column_index]) and pd.isnull(
                                    data_dictionary_out.loc[row_index, column_index])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_index} value should be: {data_dictionary_in.loc[row_index, column_index]} but is: {data_dictionary_out.loc[row_index, column_index]}")

        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field is not in the dataframe")

        elif field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            if special_type_input == SpecialType.MISSING:
                if missing_values is not None:
                    for idx, value in data_dictionary_in[field_in].items():
                        if value in missing_values or pd.isnull(value):
                            if idx == len(data_dictionary_in) - 1:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                            else:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx + 1, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx + 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx + 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:
                            if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(data_dictionary_out.loc[idx, field_out])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

                else:
                    for idx, value in data_dictionary_in[field_in].items():
                        if pd.isnull(value):
                            if idx == len(data_dictionary_in) - 1:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                            else:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx + 1, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx + 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx + 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:
                            if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(data_dictionary_out.loc[idx, field_out])):
                                keep_no_trans_result = False

            elif special_type_input == SpecialType.INVALID:
                if missing_values is not None:
                    for idx, value in data_dictionary_in[field_in].items():
                        if value in missing_values:
                            if idx == len(data_dictionary_in) - 1:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                            else:
                                if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx + 1, field_in]:
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx + 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx + 1, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else:
                            if data_dictionary_out.loc[idx, field_out] != data_dictionary_in.loc[idx, field_in] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(
                                data_dictionary_out.loc[idx, field_out])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.loc[idx, field_in]} but is: {data_dictionary_out.loc[idx, field_out]}")

    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_derived_type_col_row_outliers(derivedTypeOutput: DerivedType, data_dictionary_in: pd.DataFrame,
                                        data_dictionary_out: pd.DataFrame, outliers_dataframe_mask: pd.DataFrame,
                                        belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                        axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check the derived type to the outliers of a dataframe
    :param derivedTypeOutput: derived type to apply to the outliers
    :param data_dictionary_in: original dataframe with the data
    :param data_dictionary_out: dataframe with the derived type applied to the outliers
    :param outliers_dataframe_mask: dataframe with the outliers mask
    :param belong_op_in: belong operator condition for the if block of the invariant
    :param belong_op_out: belong operator condition for the else block of the invariant
    :param axis_param: axis to apply the derived type. If axis_param is None, the derived type is applied to the whole dataframe.
    If axis_param is 0, the derived type is applied to each column. If axis_param is 1, the derived type is applied to each row.
    :param field_in: field to apply the derived type.
    :param field_out: field to apply the derived type.

    :return: True if the derived type is applied correctly to the outliers, False otherwise
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        if derivedTypeOutput == DerivedType.MOSTFREQUENT:
            if axis_param == 0:
                for col in data_dictionary_in.columns:
                    if np.issubdtype(data_dictionary_in[col].dtype, np.number):
                        for idx, value in data_dictionary_in[col].items():
                            if outliers_dataframe_mask.at[idx, col] == 1:
                                if data_dictionary_out.at[idx, col] != data_dictionary_in[
                                    col].value_counts().idxmax():
                                    if belong_op_out == Belong.BELONG:
                                        result = False
                                        print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in[col].value_counts().idxmax()} but is: {data_dictionary_out.at[idx, col]}")
                                    elif belong_op_out == Belong.NOTBELONG:
                                        result = True
                                        print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in[col].value_counts().idxmax()} but is: {data_dictionary_out.at[idx, col]}")
                            else:
                                if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, col] and not (
                                pd.isnull(data_dictionary_in.loc[idx, col]) and pd.isnull(
                                data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, col]} but is: {data_dictionary_out.at[idx, col]}")
            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    for col in row.index:
                        if outliers_dataframe_mask.at[idx, col] == 1:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.loc[idx].mode()[0]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.loc[idx].mode()[0]} but is: {data_dictionary_out.at[idx, col]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in.loc[idx].mode()[0]} but is: {data_dictionary_out.at[idx, col]}")
                        else:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, col] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, col]) and pd.isnull(
                                data_dictionary_out.loc[idx, col])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, col]} but is: {data_dictionary_out.at[idx, col]}")

        elif derivedTypeOutput == DerivedType.PREVIOUS:
            if axis_param == 0:
                for col in data_dictionary_in.columns:
                    for idx, value in data_dictionary_in[col].items():
                        if outliers_dataframe_mask.at[idx, col] == 1 and idx != 0:
                            if data_dictionary_out.at[idx, col] != data_dictionary_out.at[idx - 1, col]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_out.at[idx - 1, col]} but is: {data_dictionary_out.at[idx, col]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_out.at[idx - 1, col]} but is: {data_dictionary_out.at[idx, col]}")
                        else:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, col] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, col]) and pd.isnull(
                                data_dictionary_out.loc[idx, col])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, col]} but is: {data_dictionary_out.at[idx, col]}")
            elif axis_param == 1:
                for idx, row in data_dictionary_in.iterrows():
                    for col in row.index:
                        if outliers_dataframe_mask.at[idx, col] == 1 and col != 0:
                            prev_col = row.index[row.index.get_loc(col) - 1]  # Get the previous column
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, prev_col]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, prev_col]} but is: {data_dictionary_out.at[idx, col]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, prev_col]} but is: {data_dictionary_out.at[idx, col]}")
                        else:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, col] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, col]) and pd.isnull(
                                data_dictionary_out.loc[idx, col])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, col]} but is: {data_dictionary_out.at[idx, col]}")

        elif derivedTypeOutput == DerivedType.NEXT:
            if axis_param == 0:
                for col in data_dictionary_in.columns:
                    for idx, value in data_dictionary_in[col].items():
                        if outliers_dataframe_mask.at[idx, col] == 1 and idx != len(data_dictionary_in) - 1:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx + 1, col] and not (pd.isnull(
                                data_dictionary_out.at[idx, col]) and pd.isnull(data_dictionary_in.at[idx + 1, col])):
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx + 1, col]} but is: {data_dictionary_out.at[idx, col]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx + 1, col]} but is: {data_dictionary_out.at[idx, col]}")
                        else:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, col] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, col]) and pd.isnull(
                                data_dictionary_out.loc[idx, col])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, col]} but is: {data_dictionary_out.at[idx, col]}")
            elif axis_param == 1:
                for col in data_dictionary_in.columns:
                    for idx, value in data_dictionary_in[col].items():
                        if outliers_dataframe_mask.at[idx, col] == 1 and col != data_dictionary_in.columns[-1]:
                            next_col = data_dictionary_in.columns[data_dictionary_in.columns.get_loc(col) + 1]
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, next_col]:
                                if belong_op_out == Belong.BELONG:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, next_col]} but is: {data_dictionary_out.at[idx, col]}")
                                elif belong_op_out == Belong.NOTBELONG:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, next_col]} but is: {data_dictionary_out.at[idx, col]}")
                        else:
                            if data_dictionary_out.at[idx, col] != data_dictionary_in.at[idx, col] and not (
                                    pd.isnull(data_dictionary_in.loc[idx, col]) and pd.isnull(
                                data_dictionary_out.loc[idx, col])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {col} value should be: {data_dictionary_in.at[idx, col]} but is: {data_dictionary_out.at[idx, col]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field is not in the dataframe")
        elif field_in in outliers_dataframe_mask.columns:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                for idx, value in data_dictionary_in[field_in].items():
                    if outliers_dataframe_mask.at[idx, field_in] == 1:
                        if data_dictionary_out.at[idx, field_out] != data_dictionary_in[field_in].value_counts().idxmax():
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in[field_in].value_counts().idxmax()} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in[field_in].value_counts().idxmax()} but is: {data_dictionary_out.at[idx, field_out]}")
                    else:
                        if data_dictionary_out.at[idx, field_out] != data_dictionary_in.at[idx, field_in] and not (
                                pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(
                            data_dictionary_out.loc[idx, field_out])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")

            elif derivedTypeOutput == DerivedType.PREVIOUS:
                for idx, value in data_dictionary_in[field_in].items():
                    if outliers_dataframe_mask.at[idx, field_in] == 1 and idx != 0:
                        if data_dictionary_out.at[idx, field_out] != data_dictionary_out.at[idx - 1, field_out]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_out.at[idx - 1, field_out]} but is: {data_dictionary_out.at[idx, field_in]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_out.at[idx - 1, field_out]} but is: {data_dictionary_out.at[idx, field_in]}")
                    else:
                        if data_dictionary_out.at[idx, field_out] != data_dictionary_in.at[idx, field_in] and not (
                                pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(
                            data_dictionary_out.loc[idx, field_out])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")

            elif derivedTypeOutput == DerivedType.NEXT:
                for idx, value in data_dictionary_in[field_in].items():
                    if outliers_dataframe_mask.at[idx, field_in] == 1 and idx != len(data_dictionary_in) - 1:
                        if data_dictionary_out.at[idx, field_out] != data_dictionary_in.at[idx + 1, field_in]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx + 1, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx + 1, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")
                    else:
                        if data_dictionary_out.at[idx, field_out] != data_dictionary_in.at[idx, field_in] and not (
                                pd.isnull(data_dictionary_in.loc[idx, field_in]) and pd.isnull(
                            data_dictionary_out.loc[idx, field_out])):
                            keep_no_trans_result = False
                            print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {data_dictionary_in.at[idx, field_in]} but is: {data_dictionary_out.at[idx, field_out]}")

    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False
