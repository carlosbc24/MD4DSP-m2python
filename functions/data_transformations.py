# Importing functions and classes from packages


import numpy as np
import pandas as pd
from helpers.auxiliar import cast_type_FixValue, find_closest_value, check_interval_condition
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType
from helpers.transform_aux import get_outliers, special_type_mean, special_type_median, special_type_closest, \
    special_type_interpolation, apply_derived_type_col_row_outliers, apply_derived_type


class DataTransformations:
    # FixValue - FixValue, FixValue - DerivedValue, FixValue - NumOp
    # Interval - FixValue, Interval - DerivedValue, Interval - NumOp
    # SpecialValue - FixValue, SpecialValue - DerivedValue, SpecialValue - NumOp

    def transform_fix_value_fix_value(self, data_dictionary: pd.DataFrame, input_values_list: list = None,
                                      output_values_list: list = None, data_type_input_list: list = None,
                                      data_type_output_list: list = None, field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the FixValue - FixValue relation
        params:
            data_dictionary: dataframe with the data
            data_type_input: data type of the input value
            fix_value_input: input value to check
            data_type_output: data type of the output value
            fix_value_output: output value to check
            field: field to execute the data transformation
        Returns:
            data_dictionary with the fix_value_input and fix_value_output values changed to the type data_type_input and data_type_output respectively
        """
        if input_values_list.__sizeof__() != output_values_list.__sizeof__():
            raise ValueError("The input and output values lists must have the same length")

        if data_type_input_list.__sizeof__() != data_type_output_list.__sizeof__():
            raise ValueError("The input and output data types lists must have the same length")

        for i in range(len(input_values_list)):
            if data_type_input_list is not None and data_type_output_list is not None:  # If the data types are specified, the transformation is performed
                # Auxiliary function that changes the values of fix_value_input and fix_value_output to the data type in data_type_input and data_type_output respectively
                input_values_list[i], output_values_list[i] = cast_type_FixValue(data_type_input=data_type_input_list[i],
                                                                                 fix_value_input=input_values_list[i],
                                                                                 data_type_output=data_type_output_list[i],
                                                                                 fix_value_output=output_values_list[i])

        # Create a dictionary to store the mapping equivalence between the input and output values
        mapping_values = {}

        for input_value in input_values_list:
            if input_value not in mapping_values:
                mapping_values[input_value] = output_values_list[input_values_list.index(input_value)]

        if field is None:
            for column_index, column_name in enumerate(data_dictionary.columns):
                for row_index, value in data_dictionary[column_name].items():
                    if value in mapping_values:
                        data_dictionary.at[row_index, column_name] = mapping_values[value]
        elif field is not None:
            if field in data_dictionary.columns:
                for row_index in data_dictionary[field].index:
                    value = data_dictionary.at[row_index, field]
                    if value in mapping_values:
                        data_dictionary.at[row_index, field] = mapping_values[value]
            elif field not in data_dictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

        return data_dictionary

    def transform_fix_value_derived_value(self, data_dictionary: pd.DataFrame, fix_value_input,
                                          derived_type_output: DerivedType,
                                          data_type_input: DataType = None, axis_param: int = None,
                                          field: str = None) -> pd.DataFrame:
        # By default, if all values are equally frequent, it is replaced by the first value.
        # Check if it should only be done for rows and columns or also for the entire dataframe.
        """
        Execute the data transformation of the FixValue - DerivedValue relation
        Sustituye el valor proporcionado por el usuario por el valor derivado en el eje que se especifique por parámetros
        params:
            data_dictionary: dataframe with the data
            data_type_input: data type of the input value
            fix_value_input: input value to check
            derived_type_output: derived type of the output value
            axis_param: axis to execute the data transformation - 0: column, None: dataframe
            field: field to execute the data transformation

            return: data_dictionary with the fix_value_input values replaced by the value derived from the operation derived_type_output
        """
        if data_type_input is not None:  # If the data type is specified, the transformation is performed
            fix_value_input, valorNulo = cast_type_FixValue(data_type_input, fix_value_input, None, None)
            # Auxiliary function that changes the value of fix_value_input to the data type in data_type_input
        data_dictionary_copy = data_dictionary.copy()

        if field is None:
            if derived_type_output == DerivedType.MOSTFREQUENT:
                if axis_param == 1:  # Applies the lambda function at the row level
                    data_dictionary_copy = data_dictionary_copy.apply(lambda fila: fila.apply(
                        lambda value: data_dictionary_copy.loc[
                            fila.name].value_counts().idxmax() if value == fix_value_input else value), axis=axis_param)
                elif axis_param == 0:  # Applies the lambda function at the column level
                    data_dictionary_copy = data_dictionary_copy.apply(lambda columna: columna.apply(
                        lambda value: data_dictionary_copy[
                            columna.name].value_counts().idxmax() if value == fix_value_input else value),
                                                                    axis=axis_param)
                else:  # Applies the lambda function at the dataframe level
                    # In case of a tie of the value with the most appearances in the dataset, the first value is taken
                    valor_mas_frecuente = data_dictionary_copy.stack().value_counts().idxmax()
                    # Replace the values within the interval with the most frequent value in the entire DataFrame using lambda
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda col: col.replace(fix_value_input, valor_mas_frecuente))
            # If it is the first value, it is replaced by the previous value in the same column
            elif derived_type_output == DerivedType.PREVIOUS:
                # Applies the lambda function at the column level or at the row level
                # Lambda that replaces any value equal to fix_value_input in the dataframe with the value of the previous position in the same column
                if axis_param is not None:
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda x: x.where((x != fix_value_input) | x.shift(1).isna(),
                                          other=x.shift(1)), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the PREVIOUS operation")
            # It assigns the value np.nan if it is the last value
            elif derived_type_output == DerivedType.NEXT:
                # Applies the lambda function at the column level or at the row level
                if axis_param is not None:
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda x: x.where((x != fix_value_input) | x.shift(-1).isna(),
                                          other=x.shift(-1)), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the NEXT operation")
        elif field is not None:
            if field not in data_dictionary.columns:
                raise ValueError("The field does not exist in the dataframe")
            elif field in data_dictionary.columns:
                if derived_type_output == DerivedType.MOSTFREQUENT:
                    data_dictionary_copy[field] = data_dictionary_copy[field].apply(lambda value: data_dictionary_copy[
                        field].value_counts().idxmax() if value == fix_value_input else value)
                elif derived_type_output == DerivedType.PREVIOUS:
                    data_dictionary_copy[field] = data_dictionary_copy[field].where(
                        (data_dictionary_copy[field] != fix_value_input) | data_dictionary_copy[field].shift(1).isna(),
                        other=data_dictionary_copy[field].shift(1))
                elif derived_type_output == DerivedType.NEXT:
                    data_dictionary_copy[field] = data_dictionary_copy[field].where(
                        (data_dictionary_copy[field] != fix_value_input) | data_dictionary_copy[field].shift(-1).isna(),
                        other=data_dictionary_copy[field].shift(-1))

        return data_dictionary_copy

    def transform_fix_value_num_op(self, data_dictionary: pd.DataFrame, fix_value_input, num_op_output: Operation,
                                   data_type_input: DataType = None, axis_param: int = None,
                                   field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the FixValue - NumOp relation
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        params:
            data_dictionary: dataframe with the data
            data_type_input: data type of the input value
            fix_value_input: input value to check
            num_op_output: operation to execute the data transformation
            axis_param: axis to execute the data transformation
            field: field to execute the data transformation
        Returns:
            data_dictionary with the fix_value_input values replaced by the result of the operation num_op_output
        """
        if data_type_input is not None:  # If it is specified, the transformation is performed
            fix_value_input, valorNulo = cast_type_FixValue(data_type_input, fix_value_input, None, None)

        # Auxiliary function that changes the value of 'fix_value_input' to the data type in 'data_type_input'
        data_dictionary_copy = data_dictionary.copy()
        if field is None:
            if num_op_output == Operation.INTERPOLATION:
                # Applies linear interpolation to the entire DataFrame
                data_dictionary_copy_copy = data_dictionary_copy.copy()
                if axis_param == 0:
                    for col in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy_copy[col].dtype, np.number):
                            # Step 1: Replace the values that meet the condition with NaN
                            data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(lambda x: np.nan if x == fix_value_input else x)
                            # Step 2: Interpolate the resulting NaN values
                            data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')

                    # Iterate over each column
                    for col in data_dictionary_copy.columns:
                        # For each index in the column
                        for idx in data_dictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(data_dictionary_copy.at[idx, col]):
                                # Replace the value with the corresponding one from data_dictionary_copy_copy
                                data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                    return data_dictionary_copy_copy
                elif axis_param == 1:
                    data_dictionary_copy_copy = data_dictionary_copy_copy.T
                    data_dictionary_copy = data_dictionary_copy.T
                    for col in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy_copy[col].dtype, np.number):
                            # Step 1: Replace the values that meet the condition with NaN
                            data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(lambda x: np.nan if x == fix_value_input else x)
                            # Step 2: Interpolate the resulting NaN values
                            data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                    # Iterate over each column
                    for col in data_dictionary_copy.columns:
                        # For each index in the column
                        for idx in data_dictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(data_dictionary_copy.at[idx, col]):
                                # Replace the value with the corresponding one from data_dictionary_copy_copy
                                data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                    data_dictionary_copy_copy = data_dictionary_copy_copy.T
                    return data_dictionary_copy_copy
                elif axis_param is None:
                    raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

            elif num_op_output == Operation.MEAN:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Replace 'fix_value_input' with the mean of the entire DataFrame using lambda
                    data_dictionary_copy = data_dictionary_copy.replace(fix_value_input, mean_value)
                elif axis_param == 0:
                    means = data_dictionary_copy.apply(
                        lambda col: col[col.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                            col.dtype, np.number) else None)

                    for col in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                            data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                                lambda x: x if x != fix_value_input else means[col])
                elif axis_param == 1:
                    data_dictionary_copy = data_dictionary_copy.T

                    means = data_dictionary_copy.apply(
                        lambda row: row[row.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                            row.dtype, np.number) else None)

                    for row in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[row].dtype, np.number):
                            data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                                lambda x: x if x != fix_value_input else means[row])

                    data_dictionary_copy = data_dictionary_copy.T

            elif num_op_output == Operation.MEDIAN:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                    # Calculate the median of these numeric columns
                    median_value = only_numbers_df.median().median()
                    # Replace 'fix_value_input' with the median of the entire DataFrame using lambda
                    data_dictionary_copy = data_dictionary_copy.replace(fix_value_input, median_value)
                elif axis_param == 0:
                    for col in data_dictionary_copy.columns:
                        if data_dictionary_copy[col].isin([fix_value_input]).any():
                            median=data_dictionary_copy[col].median()
                            data_dictionary_copy[col] = data_dictionary_copy[col].replace(fix_value_input, median)
                elif axis_param == 1:
                    data_dictionary_copy = data_dictionary_copy.T

                    for row in data_dictionary_copy.columns:
                        if data_dictionary_copy[row].isin([fix_value_input]).any():
                            median = data_dictionary_copy[row].median()
                            data_dictionary_copy[row] = data_dictionary_copy[row].replace(fix_value_input, median)

                    data_dictionary_copy = data_dictionary_copy.T

            elif num_op_output == Operation.CLOSEST:
                if axis_param is None:
                    closest_value=find_closest_value(data_dictionary_copy.stack(), fix_value_input)
                    data_dictionary_copy=data_dictionary_copy.replace(fix_value_input, closest_value)
                elif axis_param == 0:
                    # Replace 'fix_value_input' with the closest numeric value along the columns
                    for col in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[col].dtype, np.number) and data_dictionary_copy[col].isin([fix_value_input]).any():
                            closest_value=find_closest_value(data_dictionary_copy[col], fix_value_input)
                            data_dictionary_copy[col] = data_dictionary_copy[col].replace(fix_value_input, closest_value)
                elif axis_param == 1:
                    # Replace 'fix_value_input' with the closest numeric value along the rows
                    data_dictionary_copy = data_dictionary_copy.T
                    for row in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[row].dtype, np.number) and data_dictionary_copy[row].isin([fix_value_input]).any():
                            closest_value=find_closest_value(data_dictionary_copy[row], fix_value_input)
                            data_dictionary_copy[row] = data_dictionary_copy[row].replace(fix_value_input, closest_value)
                    data_dictionary_copy = data_dictionary_copy.T
            else:
                raise ValueError("No valid operator")
        elif field is not None:
            if field not in data_dictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in data_dictionary.columns:
                if np.issubdtype(data_dictionary_copy[field].dtype, np.number):
                    if num_op_output == Operation.INTERPOLATION:
                        data_dictionary_copy_copy = data_dictionary_copy.copy()
                        data_dictionary_copy_copy[field] = data_dictionary_copy_copy[field].apply(lambda x: x if x != fix_value_input else np.nan)
                        data_dictionary_copy_copy[field]=data_dictionary_copy_copy[field].interpolate(method='linear',limit_direction='both')
                        for idx in data_dictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(data_dictionary_copy.at[idx, field]):
                                # Replace the value with the corresponding one from data_dictionary_copy_copy
                                data_dictionary_copy_copy.at[idx, field] = data_dictionary_copy.at[idx, field]
                        return data_dictionary_copy_copy
                    elif num_op_output == Operation.MEAN:
                        if data_dictionary_copy[field].isin([fix_value_input]).any():
                            mean=data_dictionary_copy[field].mean()
                            data_dictionary_copy[field] = data_dictionary_copy[field].replace(fix_value_input, mean)
                    elif num_op_output == Operation.MEDIAN:
                        if data_dictionary_copy[field].isin([fix_value_input]).any():
                            median=data_dictionary_copy[field].median()
                            data_dictionary_copy[field] = data_dictionary_copy[field].replace(fix_value_input, median)
                    elif num_op_output == Operation.CLOSEST:
                        if data_dictionary_copy[field].isin([fix_value_input]).any():
                            closest_value=find_closest_value(data_dictionary_copy[field], fix_value_input)
                            data_dictionary_copy[field] = data_dictionary_copy[field].replace(fix_value_input, closest_value)
                else:
                    raise ValueError("The field is not numeric")

        return data_dictionary_copy

    def transform_interval_fix_value(self, data_dictionary: pd.DataFrame, left_margin: float, right_margin: float,
                                     closure_type: Closure, fix_value_output, data_type_output: DataType = None,
                                     field_in: str = None, field_out: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the Interval - FixValue relation
        :param data_dictionary: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param data_type_output: data type of the output value
        :param fix_value_output: output value to check
        :param field: field to execute the data transformation
        :return: data_dictionary with the values of the interval changed to the value fix_value_output
        """
        if data_type_output is not None:  # If it is specified, the transformation is performed
            vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

        data_dictionary_copy = data_dictionary.copy()

        if field_in is None:
            # Apply the lambda function to the entire dataframe
            if closure_type == Closure.openOpen:
                data_dictionary_copy = data_dictionary.apply(lambda func: func.apply(lambda x: fix_value_output if (
                                np.issubdtype(type(x), np.number) and left_margin < x < right_margin) else x))
            elif closure_type == Closure.openClosed:
                data_dictionary_copy = data_dictionary.apply(lambda func: func.apply(lambda x: fix_value_output if
                                np.issubdtype(type(x), np.number) and (left_margin < x) and (x <= right_margin) else x))
            elif closure_type == Closure.closedOpen:
                data_dictionary_copy = data_dictionary.apply(lambda func: func.apply(lambda x: fix_value_output if
                                np.issubdtype(type(x), np.number) and (left_margin <= x) and (x < right_margin) else x))
            elif closure_type == Closure.closedClosed:
                data_dictionary_copy = data_dictionary.apply(lambda func: func.apply(lambda x: fix_value_output if
                                np.issubdtype(type(x), np.number) and (left_margin <= x) and (x <= right_margin) else x))

        elif field_in is not None:
            if field_in not in data_dictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field_in in data_dictionary.columns:
                if np.issubdtype(data_dictionary[field_in].dtype, np.number):
                    if closure_type == Closure.openOpen:
                        data_dictionary_copy[field_out] = data_dictionary[field_in].apply(
                            lambda x: fix_value_output if (left_margin < x) and (x < right_margin) else x)
                    elif closure_type == Closure.openClosed:
                        data_dictionary_copy[field_out] = data_dictionary[field_in].apply(
                            lambda x: fix_value_output if (left_margin < x) and (x <= right_margin) else x)
                    elif closure_type == Closure.closedOpen:
                        data_dictionary_copy[field_out] = data_dictionary[field_in].apply(
                            lambda x: fix_value_output if (left_margin <= x) and (x < right_margin) else x)
                    elif closure_type == Closure.closedClosed:
                        data_dictionary_copy[field_out] = data_dictionary[field_in].apply(
                            lambda x: fix_value_output if (left_margin <= x) and (x <= right_margin) else x)
                else:
                    raise ValueError("The field is not numeric")

        return data_dictionary_copy

    def transform_interval_derived_value(self, data_dictionary: pd.DataFrame, left_margin: float, right_margin: float,
                                         closure_type: Closure, derived_type_output: DerivedType,
                                         axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the Interval - DerivedValue relation
        :param data_dictionary: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param derived_type_output: derived type of the output value
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation

        :return: data_dictionary with the values of the interval changed to the
            value derived from the operation derived_type_output
        """
        data_dictionary_copy = data_dictionary.copy()

        if field is None:
            if derived_type_output == DerivedType.MOSTFREQUENT:
                if axis_param == 1:  # Applies the lambda function at the row level
                    data_dictionary_copy = data_dictionary_copy.T
                    for row in data_dictionary_copy.columns:
                        most_frequent=data_dictionary_copy[row].value_counts().idxmax()
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(lambda x: most_frequent if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
                    data_dictionary_copy = data_dictionary_copy.T
                elif axis_param == 0:  # Applies the lambda function at the column level
                    for col in data_dictionary_copy.columns:
                        most_frequent = data_dictionary_copy[col].value_counts().idxmax()
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(lambda x: most_frequent if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
                else:  # Applies the lambda function at the dataframe level
                    # In case of a tie of the value with the most appearances in the dataset, the first value is taken
                    valor_mas_frecuente = data_dictionary_copy.stack().value_counts().idxmax()
                    # Replace the values within the interval with the most frequent value in the entire DataFrame using lambda
                    data_dictionary_copy = data_dictionary_copy.apply(lambda columna: columna.apply(
                        lambda value: valor_mas_frecuente if check_interval_condition(value, left_margin, right_margin, closure_type) else value))
            # Doesn't assign anything to np.nan
            elif derived_type_output == DerivedType.PREVIOUS:
                # Applies the lambda function at the column level or at the row level
                # Lambda that replaces any value within the interval in the dataframe with the value of the previous position in the same column
                if axis_param is not None:
                    # Define a lambda function to replace the values within the interval with the value of the previous position in the same column
                    data_dictionary_copy = data_dictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i - 1] if check_interval_condition(value, left_margin, right_margin, closure_type) and i > 0 else value
                                  for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the PREVIOUS operation")
            # Doesn't assign anything to np.nan
            elif derived_type_output == DerivedType.NEXT:
                # Applies the lambda function at the column level or at the row level
                if axis_param is not None:
                    # Define the lambda function to replace the values within the interval with the value of the next position in the same column
                    data_dictionary_copy = data_dictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i + 1] if check_interval_condition(value, left_margin, right_margin, closure_type) and i < len(row_or_col) - 1 else value
                               for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the NEXT operation")

        elif field is not None:
            if field not in data_dictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in data_dictionary.columns:
                if derived_type_output == DerivedType.MOSTFREQUENT:
                    most_frequent = data_dictionary_copy[field].value_counts().idxmax()
                    data_dictionary_copy[field] = data_dictionary_copy[field].apply(lambda value:
                                                most_frequent if check_interval_condition(value, left_margin, right_margin, closure_type) else value)
                elif derived_type_output == DerivedType.PREVIOUS:
                    data_dictionary_copy[field] = pd.Series(
                        [np.nan if pd.isnull(value) else data_dictionary_copy[field].iloc[i - 1]
                        if check_interval_condition(value, left_margin, right_margin, closure_type) and i > 0 else value for i, value in enumerate(data_dictionary_copy[field])],
                            index=data_dictionary_copy[field].index)
                elif derived_type_output == DerivedType.NEXT:
                    data_dictionary_copy[field] = pd.Series(
                        [np.nan if pd.isnull(value) else data_dictionary_copy[field].iloc[i + 1]
                        if check_interval_condition(value, left_margin, right_margin, closure_type) and i < len(data_dictionary_copy[field]) - 1 else value for i, value in
                         enumerate(data_dictionary_copy[field])], index=data_dictionary_copy[field].index)

        return data_dictionary_copy

    def transform_interval_num_op(self, data_dictionary: pd.DataFrame, left_margin: float, right_margin: float,
                                  closure_type: Closure, num_op_output: Operation, axis_param: int = None,
                                  field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the FixValue - NumOp relation
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        :param data_dictionary: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param num_op_output: operation to execute the data transformation
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation
        :return: data_dictionary with the values of the interval changed to the result of the operation num_op_output
        """
        # Auxiliary function that changes the value of 'fix_value_input' to the data type in 'data_type_input'
        data_dictionary_copy = data_dictionary.copy()

        if field is None:
            if num_op_output == Operation.INTERPOLATION:
                # Applies linear interpolation to the entire DataFrame
                data_dictionary_copy_copy=data_dictionary_copy.copy()
                if axis_param == 0:
                    for col in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                            data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(lambda x: np.nan if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
                            data_dictionary_copy_copy[col]=data_dictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                    # Iterate over each column
                    for col in data_dictionary_copy.columns:
                        # For each index in the column
                        for idx in data_dictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(data_dictionary_copy.at[idx, col]):
                                # Replace the value with the corresponding one from data_dictionary_copy_copy
                                data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                    return data_dictionary_copy_copy
                elif axis_param == 1:
                    data_dictionary_copy_copy = data_dictionary_copy_copy.T
                    data_dictionary_copy = data_dictionary_copy.T
                    for col in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                            data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(lambda x: np.nan if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
                            data_dictionary_copy_copy[col]=data_dictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                    # Iterate over each column
                    for col in data_dictionary_copy.columns:
                        # For each index in the column
                        for idx in data_dictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(data_dictionary_copy.at[idx, col]):
                                # Replace the value with the corresponding one from data_dictionary_copy_copy
                                data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                    data_dictionary_copy_copy = data_dictionary_copy_copy.T
                    return data_dictionary_copy_copy
                elif axis_param is None:
                    raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

            elif num_op_output == Operation.MEAN:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Replace the values within the interval with the mean of the entire DataFrame using lambda
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: mean_value if (np.issubdtype(type(x), np.number) and check_interval_condition(x, left_margin, right_margin, closure_type)) else x))
                elif axis_param == 0:
                    means = data_dictionary_copy.apply(lambda col: col[col.apply(lambda x:
                            np.issubdtype(type(x), np.number))].mean() if np.issubdtype(col.dtype, np.number) else None)
                    for col in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                            data_dictionary_copy[col] = data_dictionary_copy[col].apply(lambda x: means[col] if
                                                        check_interval_condition(x, left_margin, right_margin, closure_type) else x)
                elif axis_param == 1:
                    data_dictionary_copy = data_dictionary_copy.T
                    means = data_dictionary_copy.apply(lambda row: row[row.apply(lambda x:
                                np.issubdtype(type(x), np.number))].mean() if np.issubdtype(row.dtype, np.number) else None)
                    for row in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[row].dtype, np.number):
                            data_dictionary_copy[row] = data_dictionary_copy[row].apply(lambda x: means[row] if
                                                        check_interval_condition(x, left_margin, right_margin, closure_type) else x)

                    data_dictionary_copy = data_dictionary_copy.T

            elif num_op_output == Operation.MEDIAN:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                    # Calculate the median of these numeric columns
                    median_value = only_numbers_df.median().median()
                    # Replace the values within the interval with the median of the entire DataFrame using lambda
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: median_value if (np.issubdtype(type(x), np.number) and check_interval_condition(x, left_margin, right_margin, closure_type)) else x))

                elif axis_param == 0:
                    for col in data_dictionary_copy.select_dtypes(include=[np.number]).columns:
                        median=data_dictionary_copy[col].median()
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                            lambda x: x if not check_interval_condition(x, left_margin, right_margin, closure_type) else median)
                elif axis_param == 1:
                    data_dictionary_copy = data_dictionary_copy.T
                    for row in data_dictionary_copy.select_dtypes(include=[np.number]).columns:
                        median=data_dictionary_copy[row].median()
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                            lambda x: x if not check_interval_condition(x, left_margin, right_margin, closure_type) else median)
                    data_dictionary_copy = data_dictionary_copy.T

            elif num_op_output == Operation.CLOSEST:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                    # Flatten the dataframe into a list of values
                    flattened_values = only_numbers_df.values.flatten().tolist()
                    # Create a dictionary to store the closest value for each value in the interval
                    closest_values = {}
                    # Iterate over the values in the interval
                    for value in flattened_values:
                        # Check if the value is within the interval
                        if check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(
                                value):
                            # Check if the value is already in the dictionary
                            if value not in closest_values:
                                # Find the closest value to the current value in the interval
                                closest_values[value] = find_closest_value(flattened_values, value)

                    # Check if the closest values have been replaced in the data_dictionary_out
                    for idx, row in data_dictionary_copy.iterrows():
                        for col_name in data_dictionary_copy.columns:
                            if (np.isreal(row[col_name]) and check_interval_condition(row[col_name], left_margin,
                                        right_margin, closure_type) and not pd.isnull(row[col_name])):
                                data_dictionary_copy.at[idx, col_name] = closest_values[row[col_name]]
                elif axis_param == 0:
                    for col_name in data_dictionary_copy.select_dtypes(include=[np.number]).columns:
                        # Flatten the column into a list of values
                        flattened_values = data_dictionary_copy[col_name].values.flatten().tolist()
                        # Create a dictionary to store the closest value for each value in the interval
                        closest_values = {}
                        # Iterate over the values in the interval
                        for value in flattened_values:
                            # Check if the value is within the interval
                            if check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(
                                    value):
                                # Check if the value is already in the dictionary
                                if value not in closest_values:
                                    # Find the closest value to the current value in the interval
                                    closest_values[value] = find_closest_value(flattened_values, value)

                        # Check if the closest values have been replaced in the data_dictionary_out
                        for idx, value in data_dictionary_copy[col_name].items():
                            if check_interval_condition(data_dictionary_copy.at[idx, col_name], left_margin, right_margin,
                                                        closure_type):
                                data_dictionary_copy.at[idx, col_name] = closest_values[value]
                elif axis_param == 1:
                    for idx, row in data_dictionary_copy.iterrows():
                        # Flatten the row into a list of values
                        flattened_values = row.values.flatten().tolist()
                        # Create a dictionary to store the closest value for each value in the interval
                        closest_values = {}
                        # Iterate over the values in the interval
                        for value in flattened_values:
                            # Check if the value is within the interval
                            if np.issubdtype(value, np.number) and check_interval_condition(value, left_margin,
                                                    right_margin, closure_type) and not pd.isnull(value):
                                # Check if the value is already in the dictionary
                                if value not in closest_values:
                                    # Find the closest value to the current value in the interval
                                    closest_values[value] = find_closest_value(flattened_values, value)

                        # Check if the closest values have been replaced in the data_dictionary_out
                        for col_name, value in row.items():
                            if np.isreal(data_dictionary_copy.at[idx, col_name]) and check_interval_condition(
                                    data_dictionary_copy.at[idx, col_name], left_margin, right_margin,
                                    closure_type) and not pd.isnull(data_dictionary_copy.at[idx, col_name]):
                                data_dictionary_copy.at[idx, col_name] = closest_values[value]

        elif field is not None:
            if field not in data_dictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in data_dictionary.columns:
                if np.issubdtype(data_dictionary_copy[field].dtype, np.number):
                    if num_op_output == Operation.INTERPOLATION:
                        data_dictionary_copy_copy = data_dictionary_copy.copy()
                        data_dictionary_copy_copy[field] = data_dictionary_copy_copy[field].apply(lambda x: np.nan if check_interval_condition(x, left_margin, right_margin, closure_type) else x)
                        data_dictionary_copy_copy[field]=data_dictionary_copy_copy[field].interpolate(method='linear', limit_direction='both')
                        # For each index in the column
                        for idx in data_dictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(data_dictionary_copy.at[idx, field]):
                                # Replace the value with the corresponding one from data_dictionary_copy_copy
                                data_dictionary_copy_copy.at[idx, field] = data_dictionary_copy.at[idx, field]
                        return data_dictionary_copy_copy
                    elif num_op_output == Operation.MEAN:
                        mean=data_dictionary_copy[field].mean()
                        data_dictionary_copy[field] = data_dictionary_copy[field].apply(
                            lambda x: x if not check_interval_condition(x, left_margin, right_margin, closure_type) else mean)
                    elif num_op_output == Operation.MEDIAN:
                        median = data_dictionary_copy[field].median()
                        data_dictionary_copy[field] = data_dictionary_copy[field].apply(
                            lambda x: x if not check_interval_condition(x, left_margin, right_margin, closure_type) else median)
                    elif num_op_output == Operation.CLOSEST:
                        # Flatten the column into a list of values
                        flattened_values = data_dictionary_copy[field].values.flatten().tolist()
                        # Create a dictionary to store the closest value for each value in the interval
                        closest_values = {}
                        # Iterate over the values in the interval
                        for value in flattened_values:
                            # Check if the value is within the interval
                            if check_interval_condition(value, left_margin, right_margin, closure_type) and not pd.isnull(
                                    value):
                                # Check if the value is already in the dictionary
                                if value not in closest_values:
                                    # Find the closest value to the current value in the interval
                                    closest_values[value] = find_closest_value(flattened_values, value)

                        # Check if the closest values have been replaced in the data_dictionary_out
                        for idx, value in data_dictionary_copy[field].items():
                            if check_interval_condition(data_dictionary_copy.at[idx, field], left_margin, right_margin, closure_type):
                                data_dictionary_copy.at[idx, field] = closest_values[value]
                else:
                    raise ValueError("The field is not numeric")

        return data_dictionary_copy

    def transform_special_value_fix_value(self, data_dictionary: pd.DataFrame, special_type_input: SpecialType,
                                          fix_value_output, data_type_output: DataType = None, missing_values: list = None,
                                          axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the SpecialValue - FixValue relation
        :param data_dictionary: dataframe with the data
        :param special_type_input: special type of the input value
        :param data_type_output: data type of the output value
        :param fix_value_output: output value to check
        :param missing_values: list of missing values
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation
        :return: data_dictionary with the values of the special type changed to the value fix_value_output
        """
        if data_type_output is not None:  # If it is specified, the casting is performed
            vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

        data_dictionary_copy = data_dictionary.copy()

        if field is None:
            if special_type_input == SpecialType.MISSING:  # Include NaN values and the values in the list missing_values
                data_dictionary_copy = data_dictionary_copy.replace(np.nan, fix_value_output)
                if missing_values is not None:
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))

            elif special_type_input == SpecialType.INVALID:  # Just include the values in the list missing_values
                if missing_values is not None:
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda col: col.apply(lambda x: fix_value_output if x in missing_values else x))

            elif special_type_input == SpecialType.OUTLIER: # Replace the outliers with the value fix_value_output
                threshold = 1.5
                if axis_param is None:
                    Q1 = data_dictionary_copy.stack().quantile(0.25)
                    Q3 = data_dictionary_copy.stack().quantile(0.75)
                    IQR = Q3 - Q1
                    # Define the lower and upper bounds
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    # Identify the outliers in the dataframe
                    numeric_values = data_dictionary_copy.select_dtypes(include=[np.number])
                    outliers = (numeric_values < lower_bound) | (numeric_values > upper_bound)
                    # Replace the outliers with the value fix_value_output
                    data_dictionary_copy[outliers] = fix_value_output

                elif axis_param == 0: # Negate the condition to replace the outliers with the value fix_value_output
                    for col in data_dictionary_copy.columns:
                        if np.issubdtype(data_dictionary_copy[col], np.number):
                            data_dictionary_copy[col] = data_dictionary_copy[col].where(~((
                                        data_dictionary_copy[col] < data_dictionary_copy[col].quantile(0.25) - threshold * (
                                        data_dictionary_copy[col].quantile(0.75) - data_dictionary_copy[col].quantile(0.25))) |
                                        (data_dictionary_copy[col] > data_dictionary_copy[col].quantile(0.75) + threshold * (
                                        data_dictionary_copy[col].quantile(0.75) - data_dictionary_copy[col].quantile(0.25)))),
                                                                                other = fix_value_output)

                elif axis_param == 1: # Negate the condition to replace the outliers with the value fix_value_output
                    Q1 = data_dictionary_copy.quantile(0.25, axis="rows")
                    Q3 = data_dictionary_copy.quantile(0.75, axis="rows")
                    IQR = Q3 - Q1
                    outliers = data_dictionary_copy[
                        (data_dictionary_copy < Q1 - threshold * IQR) | (data_dictionary_copy > Q3 + threshold * IQR)]
                    for row in outliers.index:
                        data_dictionary_copy.iloc[row] = data_dictionary_copy.iloc[row].where(
                            ~((data_dictionary_copy.iloc[row] < Q1.iloc[row] - threshold * IQR.iloc[row]) |
                              (data_dictionary_copy.iloc[row] > Q3.iloc[row] + threshold * IQR.iloc[row])),
                                other=fix_value_output)

        elif field is not None:
            if field not in data_dictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in data_dictionary.columns:
                if special_type_input == SpecialType.MISSING:
                    data_dictionary_copy[field] = data_dictionary_copy[field].replace(np.nan, fix_value_output)
                    if missing_values is not None:
                        data_dictionary_copy[field] = data_dictionary_copy[field].apply(
                            lambda x: fix_value_output if x in missing_values else x)
                elif special_type_input == SpecialType.INVALID:
                    if missing_values is not None:
                        data_dictionary_copy[field] = data_dictionary_copy[field].apply(
                            lambda x: fix_value_output if x in missing_values else x)
                elif special_type_input == SpecialType.OUTLIER:
                    if np.issubdtype(data_dictionary_copy[field].dtype, np.number):
                        threshold = 1.5
                        Q1 = data_dictionary_copy[field].quantile(0.25)
                        Q3 = data_dictionary_copy[field].quantile(0.75)
                        IQR = Q3 - Q1
                        data_dictionary_copy[field] = data_dictionary_copy[field].where(
                            ~((data_dictionary_copy[field] < Q1 - threshold * IQR) |
                              (data_dictionary_copy[field] > Q3 + threshold * IQR)),
                            other=fix_value_output)
                    else:
                        raise ValueError("The field is not numeric")

        return data_dictionary_copy

    def transform_special_value_derived_value(self, data_dictionary: pd.DataFrame, special_type_input: SpecialType,
                                              derived_type_output: DerivedType, missing_values: list = None,
                                              axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the SpecialValue - DerivedValue relation
        :param data_dictionary: dataframe with the data
        :param special_type_input: special type of the input value
        :param derived_type_output: derived type of the output value
        :param missing_values: list of missing values
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation
        :return: data_dictionary with the values of the special type changed to the value derived from the operation derived_type_output
        """
        data_dictionary_copy = data_dictionary.copy()

        if field is None:
            if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
                data_dictionary_copy = apply_derived_type(special_type_input, derived_type_output, data_dictionary_copy,
                                                          missing_values, axis_param, field)

            elif special_type_input == SpecialType.OUTLIER:
                # IMPORTANT: The function getOutliers() does the same as apply_derivedTypeOutliers() but at the dataframe level.
                # If the outliers are applied at the dataframe level, previous and next cannot be applied.

                if axis_param is None:
                    data_dictionary_copy_copy = get_outliers(data_dictionary_copy, field, axis_param)
                    missing_values = data_dictionary_copy.where(data_dictionary_copy_copy == 1).stack().tolist()
                    data_dictionary_copy = apply_derived_type(special_type_input, derived_type_output, data_dictionary_copy,
                                                              missing_values, axis_param, field)
                elif axis_param == 0 or axis_param == 1:
                    data_dictionary_copy_copy = get_outliers(data_dictionary_copy, field, axis_param)
                    data_dictionary_copy = apply_derived_type_col_row_outliers(derived_type_output, data_dictionary_copy,
                                                                               data_dictionary_copy_copy, axis_param, field)

        elif field is not None:
            if field not in data_dictionary.columns:
                raise ValueError("The field does not exist in the dataframe")
            elif field in data_dictionary.columns:
                if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
                    data_dictionary_copy = apply_derived_type(special_type_input, derived_type_output, data_dictionary_copy,
                                                              missing_values, axis_param, field)
                elif special_type_input == SpecialType.OUTLIER:
                    data_dictionary_copy_copy = get_outliers(data_dictionary_copy, field, axis_param)
                    data_dictionary_copy = apply_derived_type_col_row_outliers(derived_type_output, data_dictionary_copy,
                                                                               data_dictionary_copy_copy, axis_param, field)

        return data_dictionary_copy

    def transform_special_value_num_op(self, data_dictionary: pd.DataFrame, special_type_input: SpecialType,
                                       num_op_output: Operation,
                                       missing_values: list = None, axis_param: int = None,
                                       field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the SpecialValue - NumOp relation
        :param data_dictionary: dataframe with the data
        :param special_type_input: special type of the input value
        :param num_op_output: operation to execute the data transformation
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation
        :return: data_dictionary with the values of the special type changed to the result of the operation num_op_output
        """
        data_dictionary_copy = data_dictionary.copy()
        data_dictionary_copy_mask = None

        if special_type_input == SpecialType.OUTLIER:
            data_dictionary_copy_mask = get_outliers(data_dictionary_copy, field, axis_param)

        if num_op_output == Operation.INTERPOLATION:
            data_dictionary_copy = special_type_interpolation(data_dictionary_copy=data_dictionary_copy,
                                                              special_type_input=special_type_input,
                                                              data_dictionary_copy_mask=data_dictionary_copy_mask,
                                                              missing_values=missing_values, axis_param=axis_param,
                                                              field=field)
        elif num_op_output == Operation.MEAN:
            data_dictionary_copy = special_type_mean(data_dictionary_copy=data_dictionary_copy,
                                                     special_type_input=special_type_input,
                                                     data_dictionary_copy_mask=data_dictionary_copy_mask,
                                                     missing_values=missing_values, axis_param=axis_param, field=field)
        elif num_op_output == Operation.MEDIAN:
            data_dictionary_copy = special_type_median(data_dictionary_copy=data_dictionary_copy,
                                                       special_type_input=special_type_input,
                                                       data_dictionary_copy_mask=data_dictionary_copy_mask,
                                                       missing_values=missing_values, axis_param=axis_param, field=field)
        elif num_op_output == Operation.CLOSEST:
            data_dictionary_copy = special_type_closest(data_dictionary_copy=data_dictionary_copy,
                                                        special_type_input=special_type_input,
                                                        data_dictionary_copy_mask=data_dictionary_copy_mask,
                                                        missing_values=missing_values, axis_param=axis_param, field=field)

        return data_dictionary_copy


    def transform_derived_field(self, data_dictionary: pd.DataFrame, field_in: str, field_out: str,
                                data_type_output: DataType = None) -> pd.DataFrame:
        """
        Execute the data transformation of the DerivedField relation
        Args:
            data_dictionary: dataframe with the data
            data_type_output: data type of the output field
            field_in: field to execute the data transformation
            field_out: field to store the output value
        Returns:
            pd.DataFrame:
        """
        data_dictionary_copy=data_dictionary.copy()

        if field_in not in data_dictionary.columns:
            raise ValueError("The field does not exist in the dataframe")
        def cast_type_column():
            if data_type_output is not None:
                if data_type_output == DataType.STRING:
                    data_dictionary_copy[field_out] = data_dictionary_copy[field_out].astype(str)
                elif data_type_output == DataType.TIME:
                    data_dictionary_copy[field_out] = data_dictionary_copy[field_out].astype('datetime64[ns]')
                elif data_type_output == DataType.INTEGER:
                    data_dictionary_copy[field_out] = data_dictionary_copy[field_out].astype(int)
                elif data_type_output == DataType.DATETIME:
                    data_dictionary_copy[field_out] = data_dictionary_copy[field_out].astype('datetime64[ns]')
                elif data_type_output == DataType.BOOLEAN:
                    data_dictionary_copy[field_out] = data_dictionary_copy[field_out].astype(bool)
                elif data_type_output == DataType.DOUBLE or data_type_output == DataType.FLOAT:
                    data_dictionary_copy[field_out] = data_dictionary_copy[field_out].astype(float)

        data_dictionary_copy[field_out] = data_dictionary_copy[field_in].copy()
        if data_type_output is not None:
            cast_type_column()

        return data_dictionary_copy