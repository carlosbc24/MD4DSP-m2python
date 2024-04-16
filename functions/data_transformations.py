# Importing functions and classes from packages
import numpy as np
import pandas as pd
from helpers.auxiliar import cast_type_FixValue, find_closest_value
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType
from helpers.transform_aux import getOutliers, specialTypeMean, specialTypeMedian, specialTypeClosest, \
    specialTypeInterpolation, apply_derivedTypeColRowOutliers, apply_derivedType


class Invariants:
    # FixValue - FixValue, FixValue - DerivedValue, FixValue - NumOp
    # Interval - FixValue, Interval - DerivedValue, Interval - NumOp
    # SpecialValue - FixValue, SpecialValue - DerivedValue, SpecialValue - NumOp

    def transform_FixValue_FixValue(self, dataDictionary: pd.DataFrame, fixValueInput, fixValueOutput,
                                    dataTypeInput: DataType = None,
                                    dataTypeOutput: DataType = None, field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the FixValue - FixValue relation
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            dataTypeOutput: data type of the output value
            FixValueOutput: output value to check
            field: field to execute the data transformation
        Returns:
            dataDictionary with the FixValueInput and FixValueOutput values changed to the type dataTypeInput and dataTypeOutput respectively
        """
        if dataTypeInput is not None and dataTypeOutput is not None:  # If the data types are specified, the transformation is performed
            # Auxiliary function that changes the values of FixValueInput and FixValueOutput to the data type in DataTypeInput and DataTypeOutput respectively
            fixValueInput, fixValueOutput = cast_type_FixValue(dataTypeInput, fixValueInput, dataTypeOutput,
                                                               fixValueOutput)

        if field is None:
            dataDictionary = dataDictionary.replace(fixValueInput, fixValueOutput)
        elif field is not None:
            if field in dataDictionary.columns:
                dataDictionary.loc[:, field] = dataDictionary.loc[:, field].replace(fixValueInput, fixValueOutput)
                # dataDictionary[field] = dataDictionary[field].replace(fixValueInput, fixValueOutput)
            elif field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

        return dataDictionary

    def transform_FixValue_DerivedValue(self, dataDictionary: pd.DataFrame, fixValueInput,
                                        derivedTypeOutput: DerivedType,
                                        dataTypeInput: DataType = None, axis_param: int = None,
                                        field: str = None) -> pd.DataFrame:
        # By default, if all values are equally frequent, it is replaced by the first value.
        # Check if it should only be done for rows and columns or also for the entire dataframe.
        """
        Execute the data transformation of the FixValue - DerivedValue relation
        Sustituye el valor proporcionado por el usuario por el valor derivado en el eje que se especifique por parámetros
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            derivedTypeOutput: derived type of the output value
            axis_param: axis to execute the data transformation - 0: column, None: dataframe
            field: field to execute the data transformation

            return: dataDictionary with the FixValueInput values replaced by the value derived from the operation derivedTypeOutput
        """
        if dataTypeInput is not None:  # If the data type is specified, the transformation is performed
            fixValueInput, valorNulo = cast_type_FixValue(dataTypeInput, fixValueInput, None, None)
            # Auxiliary function that changes the value of FixValueInput to the data type in DataTypeInput
        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                if axis_param == 1:  # Applies the lambda function at the row level
                    dataDictionary_copy = dataDictionary_copy.apply(lambda fila: fila.apply(
                        lambda value: dataDictionary_copy.loc[
                            fila.name].value_counts().idxmax() if value == fixValueInput else value), axis=axis_param)
                elif axis_param == 0:  # Applies the lambda function at the column level
                    dataDictionary_copy = dataDictionary_copy.apply(lambda columna: columna.apply(
                        lambda value: dataDictionary_copy[
                            columna.name].value_counts().idxmax() if value == fixValueInput else value),
                                                                    axis=axis_param)
                else:  # Applies the lambda function at the dataframe level
                    # In case of a tie of the value with the most appearances in the dataset, the first value is taken
                    valor_mas_frecuente = dataDictionary_copy.stack().value_counts().idxmax()
                    # Replace the values within the interval with the most frequent value in the entire DataFrame using lambda
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.replace(fixValueInput, valor_mas_frecuente))
            # If it is the first value, it is replaced by the previous value in the same column
            elif derivedTypeOutput == DerivedType.PREVIOUS:
                # Applies the lambda function at the column level or at the row level
                # Lambda that replaces any value equal to FixValueInput in the dataframe with the value of the previous position in the same column
                if axis_param is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda x: x.where((x != fixValueInput) | x.shift(1).isna(),
                                          other=x.shift(1)), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the PREVIOUS operation")
            # It assigns the value np.nan if it is the last value
            elif derivedTypeOutput == DerivedType.NEXT:
                # Applies the lambda function at the column level or at the row level
                if axis_param is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda x: x.where((x != fixValueInput) | x.shift(-1).isna(),
                                          other=x.shift(-1)), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the NEXT operation")
        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")
            elif field in dataDictionary.columns:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(lambda value: dataDictionary_copy[
                        field].value_counts().idxmax() if value == fixValueInput else value)
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    dataDictionary_copy[field] = dataDictionary_copy[field].where(
                        (dataDictionary_copy[field] != fixValueInput) | dataDictionary_copy[field].shift(1).isna(),
                        other=dataDictionary_copy[field].shift(1))
                elif derivedTypeOutput == DerivedType.NEXT:
                    dataDictionary_copy[field] = dataDictionary_copy[field].where(
                        (dataDictionary_copy[field] != fixValueInput) | dataDictionary_copy[field].shift(-1).isna(),
                        other=dataDictionary_copy[field].shift(-1))

        return dataDictionary_copy

    def transform_FixValue_NumOp(self, dataDictionary: pd.DataFrame, fixValueInput, numOpOutput: Operation,
                                 dataTypeInput: DataType = None, axis_param: int = None,
                                 field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the FixValue - NumOp relation
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            numOpOutput: operation to execute the data transformation
            axis_param: axis to execute the data transformation
            field: field to execute the data transformation
        Returns:
            dataDictionary with the FixValueInput values replaced by the result of the operation numOpOutput
        """
        if dataTypeInput is not None:  # If it is specified, the transformation is performed
            fixValueInput, valorNulo = cast_type_FixValue(dataTypeInput, fixValueInput, None, None)

        # Auxiliary function that changes the value of 'FixValueInput' to the data type in 'DataTypeInput'
        dataDictionary_copy = dataDictionary.copy()
        if field is None:
            if numOpOutput == Operation.INTERPOLATION:
                # Applies linear interpolation to the entire DataFrame
                dataDictionary_copy_copy = dataDictionary_copy.copy()
                if axis_param == 0:
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy_copy[col].dtype, np.number):
                            # Step 1: Replace the values that meet the condition with NaN
                            dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].apply(lambda x: np.nan if x == fixValueInput else x)
                            # Step 2: Interpolate the resulting NaN values
                            dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')

                    # Iterate over each column
                    for col in dataDictionary_copy.columns:
                        # For each index in the column
                        for idx in dataDictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(dataDictionary_copy.at[idx, col]):
                                # Replace the value with the corresponding one from dataDictionary_copy_copy
                                dataDictionary_copy_copy.at[idx, col] = dataDictionary_copy.at[idx, col]
                    return dataDictionary_copy_copy
                elif axis_param == 1:
                    dataDictionary_copy_copy = dataDictionary_copy_copy.T
                    dataDictionary_copy = dataDictionary_copy.T
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy_copy[col].dtype, np.number):
                            # Step 1: Replace the values that meet the condition with NaN
                            dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].apply(lambda x: np.nan if x == fixValueInput else x)
                            # Step 2: Interpolate the resulting NaN values
                            dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                    # Iterate over each column
                    for col in dataDictionary_copy.columns:
                        # For each index in the column
                        for idx in dataDictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(dataDictionary_copy.at[idx, col]):
                                # Replace the value with the corresponding one from dataDictionary_copy_copy
                                dataDictionary_copy_copy.at[idx, col] = dataDictionary_copy.at[idx, col]
                    dataDictionary_copy_copy = dataDictionary_copy_copy.T
                    return dataDictionary_copy_copy
                elif axis_param is None:
                    raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

            elif numOpOutput == Operation.MEAN:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Replace 'fixValueInput' with the mean of the entire DataFrame using lambda
                    dataDictionary_copy = dataDictionary_copy.replace(fixValueInput, mean_value)
                elif axis_param == 0:
                    means = dataDictionary_copy.apply(
                        lambda col: col[col.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                            col.dtype, np.number) else None)

                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                            dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                                lambda x: x if x != fixValueInput else means[col])
                elif axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.T

                    means = dataDictionary_copy.apply(
                        lambda row: row[row.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                            row.dtype, np.number) else None)

                    for row in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                            dataDictionary_copy[row] = dataDictionary_copy[row].apply(
                                lambda x: x if x != fixValueInput else means[row])

                    dataDictionary_copy = dataDictionary_copy.T

            elif numOpOutput == Operation.MEDIAN:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    # Calculate the median of these numeric columns
                    median_value = only_numbers_df.median().median()
                    # Replace 'fixValueInput' with the median of the entire DataFrame using lambda
                    dataDictionary_copy = dataDictionary_copy.replace(fixValueInput, median_value)
                elif axis_param == 0:
                    for col in dataDictionary_copy.columns:
                        if dataDictionary_copy[col].isin([fixValueInput]).any():
                            median=dataDictionary_copy[col].median()
                            dataDictionary_copy[col] = dataDictionary_copy[col].replace(fixValueInput, median)
                elif axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.T

                    for row in dataDictionary_copy.columns:
                        if dataDictionary_copy[row].isin([fixValueInput]).any():
                            median = dataDictionary_copy[row].median()
                            dataDictionary_copy[row] = dataDictionary_copy[row].replace(fixValueInput, median)

                    dataDictionary_copy = dataDictionary_copy.T

            elif numOpOutput == Operation.CLOSEST:
                if axis_param is None:
                    closest_value=find_closest_value(dataDictionary_copy.stack(), fixValueInput)
                    dataDictionary_copy=dataDictionary_copy.replace(fixValueInput, closest_value)
                elif axis_param == 0:
                    # Replace 'fixValueInput' with the closest numeric value along the columns
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col].dtype, np.number) and dataDictionary_copy[col].isin([fixValueInput]).any():
                            closest_value=find_closest_value(dataDictionary_copy[col], fixValueInput)
                            dataDictionary_copy[col] = dataDictionary_copy[col].replace(fixValueInput, closest_value)
                elif axis_param == 1:
                    # Replace 'fixValueInput' with the closest numeric value along the rows
                    dataDictionary_copy = dataDictionary_copy.T
                    for row in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[row].dtype, np.number) and dataDictionary_copy[row].isin([fixValueInput]).any():
                            closest_value=find_closest_value(dataDictionary_copy[row], fixValueInput)
                            dataDictionary_copy[row] = dataDictionary_copy[row].replace(fixValueInput, closest_value)
                    dataDictionary_copy = dataDictionary_copy.T
            else:
                raise ValueError("No valid operator")
        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                    if numOpOutput == Operation.INTERPOLATION:
                        dataDictionary_copy_copy = dataDictionary_copy.copy()
                        dataDictionary_copy_copy[field] = dataDictionary_copy_copy[field].apply(lambda x: x if x != fixValueInput else np.nan)
                        dataDictionary_copy_copy[field]=dataDictionary_copy_copy[field].interpolate(method='linear',limit_direction='both')
                        for idx in dataDictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(dataDictionary_copy.at[idx, field]):
                                # Replace the value with the corresponding one from dataDictionary_copy_copy
                                dataDictionary_copy_copy.at[idx, field] = dataDictionary_copy.at[idx, field]
                        return dataDictionary_copy_copy
                    elif numOpOutput == Operation.MEAN:
                        if dataDictionary_copy[field].isin([fixValueInput]).any():
                            mean=dataDictionary_copy[field].mean()
                            dataDictionary_copy[field] = dataDictionary_copy[field].replace(fixValueInput, mean)
                    elif numOpOutput == Operation.MEDIAN:
                        if dataDictionary_copy[field].isin([fixValueInput]).any():
                            median=dataDictionary_copy[field].median()
                            dataDictionary_copy[field] = dataDictionary_copy[field].replace(fixValueInput, median)
                    elif numOpOutput == Operation.CLOSEST:
                        if dataDictionary_copy[field].isin([fixValueInput]).any():
                            closest_value=find_closest_value(dataDictionary_copy[field], fixValueInput)
                            dataDictionary_copy[field] = dataDictionary_copy[field].replace(fixValueInput, closest_value)
                else:
                    raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def transform_Interval_FixValue(self, dataDictionary: pd.DataFrame, leftMargin: float, rightMargin: float,
                                    closureType: Closure, fixValueOutput, dataTypeOutput: DataType = None,
                                    field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the Interval - FixValue relation
        :param dataDictionary: dataframe with the data
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param closureType: closure type of the interval
        :param dataTypeOutput: data type of the output value
        :param fixValueOutput: output value to check
        :param field: field to execute the data transformation
        :return: dataDictionary with the values of the interval changed to the value fixValueOutput
        """
        if dataTypeOutput is not None:  # If it is specified, the transformation is performed
            vacio, fixValueOutput = cast_type_FixValue(None, None, dataTypeOutput, fixValueOutput)

        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            # Apply the lambda function to the entire dataframe
            if closureType == Closure.openOpen:
                dataDictionary_copy = dataDictionary.apply(lambda func: func.apply(lambda x: fixValueOutput if (
                                np.issubdtype(type(x), np.number) and leftMargin < x < rightMargin) else x))
            elif closureType == Closure.openClosed:
                dataDictionary_copy = dataDictionary.apply(lambda func: func.apply(lambda x: fixValueOutput if
                                np.issubdtype(type(x), np.number) and (leftMargin < x) and (x <= rightMargin) else x))
            elif closureType == Closure.closedOpen:
                dataDictionary_copy = dataDictionary.apply(lambda func: func.apply(lambda x: fixValueOutput if
                                np.issubdtype(type(x), np.number) and (leftMargin <= x) and (x < rightMargin) else x))
            elif closureType == Closure.closedClosed:
                dataDictionary_copy = dataDictionary.apply(lambda func: func.apply(lambda x: fixValueOutput if
                                np.issubdtype(type(x), np.number) and (leftMargin <= x) and (x <= rightMargin) else x))

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if np.issubdtype(dataDictionary[field].dtype, np.number):
                    if closureType == Closure.openOpen:
                        dataDictionary_copy[field] = dataDictionary[field].apply(
                            lambda x: fixValueOutput if (leftMargin < x) and (x < rightMargin) else x)
                    elif closureType == Closure.openClosed:
                        dataDictionary_copy[field] = dataDictionary[field].apply(
                            lambda x: fixValueOutput if (leftMargin < x) and (x <= rightMargin) else x)
                    elif closureType == Closure.closedOpen:
                        dataDictionary_copy[field] = dataDictionary[field].apply(
                            lambda x: fixValueOutput if (leftMargin <= x) and (x < rightMargin) else x)
                    elif closureType == Closure.closedClosed:
                        dataDictionary_copy[field] = dataDictionary[field].apply(
                            lambda x: fixValueOutput if (leftMargin <= x) and (x <= rightMargin) else x)
                else:
                    raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def transform_Interval_DerivedValue(self, dataDictionary: pd.DataFrame, leftMargin: float, rightMargin: float,
                                        closureType: Closure, derivedTypeOutput: DerivedType,
                                        axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the Interval - DerivedValue relation
        :param dataDictionary: dataframe with the data
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param closureType: closure type of the interval
        :param derivedTypeOutput: derived type of the output value
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation

        :return: dataDictionary with the values of the interval changed to the
            value derived from the operation derivedTypeOutput
        """
        dataDictionary_copy = dataDictionary.copy()

        # Define the interval condition according to the closure type
        def get_condition(x):
            if closureType == Closure.openOpen:
                return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x < rightMargin)) else False
            elif closureType == Closure.openClosed:
                return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x <= rightMargin)) else False
            elif closureType == Closure.closedOpen:
                return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x < rightMargin)) else False
            elif closureType == Closure.closedClosed:
                return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x <= rightMargin)) else False

        if field is None:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                if axis_param == 1:  # Applies the lambda function at the row level
                    dataDictionary_copy = dataDictionary_copy.T
                    for row in dataDictionary_copy.columns:
                        most_frequent=dataDictionary_copy[row].value_counts().idxmax()
                        dataDictionary_copy[row] = dataDictionary_copy[row].apply(lambda x: most_frequent if get_condition(x) else x)
                    dataDictionary_copy = dataDictionary_copy.T
                elif axis_param == 0:  # Applies the lambda function at the column level
                    for col in dataDictionary_copy.columns:
                        most_frequent = dataDictionary_copy[col].value_counts().idxmax()
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(lambda x: most_frequent if get_condition(x) else x)
                else:  # Applies the lambda function at the dataframe level
                    # In case of a tie of the value with the most appearances in the dataset, the first value is taken
                    valor_mas_frecuente = dataDictionary_copy.stack().value_counts().idxmax()
                    # Replace the values within the interval with the most frequent value in the entire DataFrame using lambda
                    dataDictionary_copy = dataDictionary_copy.apply(lambda columna: columna.apply(
                        lambda value: valor_mas_frecuente if get_condition(value) else value))
            # Doesn't assign anything to np.nan
            elif derivedTypeOutput == DerivedType.PREVIOUS:
                # Applies the lambda function at the column level or at the row level
                # Lambda that replaces any value within the interval in the dataframe with the value of the previous position in the same column
                if axis_param is not None:
                    # Define a lambda function to replace the values within the interval with the value of the previous position in the same column
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i - 1] if get_condition(value) and i > 0 else value
                                  for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the PREVIOUS operation")
            # Doesn't assign anything to np.nan
            elif derivedTypeOutput == DerivedType.NEXT:
                # Applies the lambda function at the column level or at the row level
                if axis_param is not None:
                    # Define the lambda function to replace the values within the interval with the value of the next position in the same column
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i + 1] if get_condition(value) and i < len(row_or_col) - 1 else value
                               for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the NEXT operation")

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    most_frequent = dataDictionary_copy[field].value_counts().idxmax()
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(lambda value:
                                                most_frequent if get_condition(value) else value)
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    dataDictionary_copy[field] = pd.Series(
                        [np.nan if pd.isnull(value) else dataDictionary_copy[field].iloc[i - 1]
                        if get_condition(value) and i > 0 else value for i, value in enumerate(dataDictionary_copy[field])],
                            index=dataDictionary_copy[field].index)
                elif derivedTypeOutput == DerivedType.NEXT:
                    dataDictionary_copy[field] = pd.Series(
                        [np.nan if pd.isnull(value) else dataDictionary_copy[field].iloc[i + 1]
                        if get_condition(value) and i < len(dataDictionary_copy[field]) - 1 else value for i, value in
                         enumerate(dataDictionary_copy[field])], index=dataDictionary_copy[field].index)

        return dataDictionary_copy

    def transform_Interval_NumOp(self, dataDictionary: pd.DataFrame, leftMargin: float, rightMargin: float,
                                 closureType: Closure, numOpOutput: Operation, axis_param: int = None,
                                 field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the FixValue - NumOp relation
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        :param dataDictionary: dataframe with the data
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param closureType: closure type of the interval
        :param numOpOutput: operation to execute the data transformation
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation
        :return: dataDictionary with the values of the interval changed to the result of the operation numOpOutput
        """

        def get_condition(x):
            if closureType == Closure.openOpen:
                return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x < rightMargin)) else False
            elif closureType == Closure.openClosed:
                return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x <= rightMargin)) else False
            elif closureType == Closure.closedOpen:
                return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x < rightMargin)) else False
            elif closureType == Closure.closedClosed:
                return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x <= rightMargin)) else False

        # Auxiliary function that changes the value of 'FixValueInput' to the data type in 'DataTypeInput'
        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if numOpOutput == Operation.INTERPOLATION:
                # Applies linear interpolation to the entire DataFrame
                dataDictionary_copy_copy=dataDictionary_copy.copy()
                if axis_param == 0:
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                            dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].apply(lambda x: np.nan if get_condition(x) else x)
                            dataDictionary_copy_copy[col]=dataDictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                    # Iterate over each column
                    for col in dataDictionary_copy.columns:
                        # For each index in the column
                        for idx in dataDictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(dataDictionary_copy.at[idx, col]):
                                # Replace the value with the corresponding one from dataDictionary_copy_copy
                                dataDictionary_copy_copy.at[idx, col] = dataDictionary_copy.at[idx, col]
                    return dataDictionary_copy_copy
                elif axis_param == 1:
                    dataDictionary_copy_copy = dataDictionary_copy_copy.T
                    dataDictionary_copy = dataDictionary_copy.T
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                            dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].apply(lambda x: np.nan if get_condition(x) else x)
                            dataDictionary_copy_copy[col]=dataDictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                    # Iterate over each column
                    for col in dataDictionary_copy.columns:
                        # For each index in the column
                        for idx in dataDictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(dataDictionary_copy.at[idx, col]):
                                # Replace the value with the corresponding one from dataDictionary_copy_copy
                                dataDictionary_copy_copy.at[idx, col] = dataDictionary_copy.at[idx, col]
                    dataDictionary_copy_copy = dataDictionary_copy_copy.T
                    return dataDictionary_copy_copy
                elif axis_param is None:
                    raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

            elif numOpOutput == Operation.MEAN:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    # Calculate the mean of these numeric columns
                    mean_value = only_numbers_df.mean().mean()
                    # Replace the values within the interval with the mean of the entire DataFrame using lambda
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: mean_value if (np.issubdtype(type(x), np.number) and get_condition(x)) else x))
                elif axis_param == 0:
                    means = dataDictionary_copy.apply(lambda col: col[col.apply(lambda x:
                            np.issubdtype(type(x), np.number))].mean() if np.issubdtype(col.dtype, np.number) else None)
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                            dataDictionary_copy[col] = dataDictionary_copy[col].apply(lambda x: means[col] if
                                                        get_condition(x) else x)
                elif axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.T
                    means = dataDictionary_copy.apply(lambda row: row[row.apply(lambda x:
                                np.issubdtype(type(x), np.number))].mean() if np.issubdtype(row.dtype, np.number) else None)
                    for row in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                            dataDictionary_copy[row] = dataDictionary_copy[row].apply(lambda x: means[row] if
                                                        get_condition(x) else x)

                    dataDictionary_copy = dataDictionary_copy.T

            elif numOpOutput == Operation.MEDIAN:
                if axis_param is None:
                    # Select only columns with numeric data, including all numeric types (int, float, etc.)
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    # Calculate the median of these numeric columns
                    median_value = only_numbers_df.median().median()
                    # Replace the values within the interval with the median of the entire DataFrame using lambda
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: median_value if (np.issubdtype(type(x), np.number) and get_condition(x)) else x))

                elif axis_param == 0:
                    for col in dataDictionary_copy.columns:
                        median=dataDictionary_copy[col].median()
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                            lambda x: x if not get_condition(x) else median)
                elif axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.T
                    for row in dataDictionary_copy.columns:
                        median=dataDictionary_copy[row].median()
                        dataDictionary_copy[row] = dataDictionary_copy[row].apply(
                            lambda x: x if not get_condition(x) else median)
                    dataDictionary_copy = dataDictionary_copy.T

            elif numOpOutput == Operation.CLOSEST:
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                if axis_param is None:
                    indice_row=[]
                    indice_col=[]
                    values=[]
                    for col in only_numbers_df.columns:
                        for index, row in only_numbers_df.iterrows():
                            if get_condition(row[col]):
                                indice_row.append(index)
                                indice_col.append(col)
                                values.append(row[col])

                    if values.__len__()>0 and values is not None:
                        processed=[values[0]]
                        closest_processed=[]
                        closest_value=find_closest_value(only_numbers_df.stack(), values[0])
                        closest_processed.append(closest_value)
                        for i in range(len(values)):
                            if values[i] not in processed:
                                closest_value=find_closest_value(only_numbers_df.stack(), values[i])
                                closest_processed.append(closest_value)
                                processed.append(values[i])

                        # Iterate over each index and column
                        for i in range(len(dataDictionary_copy.index)):
                            for j in range(len(dataDictionary_copy.columns)):
                                # Get the current value
                                current_value = dataDictionary_copy.iat[i, j]
                                # Verify if the value it's in the list of values to replace
                                if current_value in processed:
                                    # Get the index of the value in the list
                                    replace_index = processed.index(current_value)
                                    # Get the nearest value to replace
                                    closest_value = closest_processed[replace_index]
                                    # Replace the value in the dataframe
                                    dataDictionary_copy.iat[i, j] = closest_value
                else:
                    if axis_param == 1:
                        dataDictionary_copy = dataDictionary_copy.T
                    # Replace the values within the interval with the closest numeric value along the columns
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    for col in only_numbers_df.columns:
                        indice_row = []
                        indice_col = []
                        values = []
                        processed = []
                        closest_processed = []

                        for index, value in only_numbers_df[col].items():
                            if get_condition(value):
                                indice_row.append(index)
                                indice_col.append(col)
                                values.append(value)

                        if len(values) > 0 and values is not None:
                            processed.append(values[0])
                            closest_processed.append(find_closest_value(only_numbers_df[col], values[0]))

                            for i in range(1, len(values)):
                                if values[i] not in processed:
                                    closest_value = find_closest_value(only_numbers_df[col], values[i])
                                    processed.append(values[i])
                                    closest_processed.append(closest_value)

                            for i, index in enumerate(indice_row):
                                dataDictionary_copy.at[index, col] = closest_processed[processed.index(values[i])]
                    if axis_param == 1:
                        dataDictionary_copy = dataDictionary_copy.T
            else:
                raise ValueError("No valid operator")

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                    if numOpOutput == Operation.INTERPOLATION:
                        dataDictionary_copy_copy = dataDictionary_copy.copy()
                        dataDictionary_copy_copy[field] = dataDictionary_copy_copy[field].apply(lambda x: np.nan if get_condition(x) else x)
                        dataDictionary_copy_copy[field]=dataDictionary_copy_copy[field].interpolate(method='linear', limit_direction='both')
                        # For each index in the column
                        for idx in dataDictionary_copy.index:
                            # Verify if the value is NaN in the original dataframe
                            if pd.isnull(dataDictionary_copy.at[idx, field]):
                                # Replace the value with the corresponding one from dataDictionary_copy_copy
                                dataDictionary_copy_copy.at[idx, field] = dataDictionary_copy.at[idx, field]
                        return dataDictionary_copy_copy
                    elif numOpOutput == Operation.MEAN:
                        mean=dataDictionary_copy[field].mean()
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if not get_condition(x) else mean)
                    elif numOpOutput == Operation.MEDIAN:
                        median = dataDictionary_copy[field].median()
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if not get_condition(x) else median)
                    elif numOpOutput == Operation.CLOSEST:
                        indice_row = []
                        values = []
                        processed = []
                        closest_processed = []

                        for index, value in dataDictionary_copy[field].items():
                            if get_condition(value):
                                indice_row.append(index)
                                values.append(value)
                        if len(values) > 0 and values is not None:
                            processed.append(values[0])
                            closest_processed.append(find_closest_value(dataDictionary_copy[field], values[0]))
                            for i in range(1, len(values)):
                                if values[i] not in processed:
                                    closest_value = find_closest_value(dataDictionary_copy[field], values[i])
                                    processed.append(values[i])
                                    closest_processed.append(closest_value)
                            for i, index in enumerate(indice_row):
                                dataDictionary_copy.at[index, field] = closest_processed[processed.index(values[i])]
                else:
                    raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def transform_SpecialValue_FixValue(self, dataDictionary: pd.DataFrame, specialTypeInput: SpecialType,
                                        fixValueOutput, dataTypeOutput: DataType = None, missing_values: list = None,
                                        axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the SpecialValue - FixValue relation
        :param dataDictionary: dataframe with the data
        :param specialTypeInput: special type of the input value
        :param dataTypeOutput: data type of the output value
        :param fixValueOutput: output value to check
        :param missing_values: list of missing values
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation
        :return: dataDictionary with the values of the special type changed to the value fixValueOutput
        """
        if dataTypeOutput is not None:  # If it is specified, the casting is performed
            vacio, fixValueOutput = cast_type_FixValue(None, None, dataTypeOutput, fixValueOutput)

        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if specialTypeInput == SpecialType.MISSING:  # Include NaN values and the values in the list missing_values
                dataDictionary_copy = dataDictionary_copy.replace(np.nan, fixValueOutput)
                if missing_values is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))

            elif specialTypeInput == SpecialType.INVALID:  # Just include the values in the list missing_values
                if missing_values is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))

            elif specialTypeInput == SpecialType.OUTLIER: # Replace the outliers with the value fixValueOutput
                threshold = 1.5
                if axis_param is None:
                    Q1 = dataDictionary_copy.stack().quantile(0.25)
                    Q3 = dataDictionary_copy.stack().quantile(0.75)
                    IQR = Q3 - Q1
                    # Define the lower and upper bounds
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    # Identify the outliers in the dataframe
                    numeric_values = dataDictionary_copy.select_dtypes(include=[np.number])
                    outliers = (numeric_values < lower_bound) | (numeric_values > upper_bound)
                    # Replace the outliers with the value fixValueOutput
                    dataDictionary_copy[outliers] = fixValueOutput

                elif axis_param == 0: # Negate the condition to replace the outliers with the value fixValueOutput
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col], np.number):
                            dataDictionary_copy[col] = dataDictionary_copy[col].where(~((
                                        dataDictionary_copy[col] < dataDictionary_copy[col].quantile(0.25) - threshold * (
                                        dataDictionary_copy[col].quantile(0.75) - dataDictionary_copy[col].quantile(0.25))) |
                                        (dataDictionary_copy[col] > dataDictionary_copy[col].quantile(0.75) + threshold * (
                                        dataDictionary_copy[col].quantile(0.75) - dataDictionary_copy[col].quantile(0.25)))),
                                                                                other = fixValueOutput)

                elif axis_param == 1: # Negate the condition to replace the outliers with the value fixValueOutput
                    Q1 = dataDictionary_copy.quantile(0.25, axis="rows")
                    Q3 = dataDictionary_copy.quantile(0.75, axis="rows")
                    IQR = Q3 - Q1
                    outliers = dataDictionary_copy[
                        (dataDictionary_copy < Q1 - threshold * IQR) | (dataDictionary_copy > Q3 + threshold * IQR)]
                    for row in outliers.index:
                        dataDictionary_copy.iloc[row] = dataDictionary_copy.iloc[row].where(
                            ~((dataDictionary_copy.iloc[row] < Q1.iloc[row] - threshold * IQR.iloc[row]) |
                              (dataDictionary_copy.iloc[row] > Q3.iloc[row] + threshold * IQR.iloc[row])),
                                other=fixValueOutput)

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy[field] = dataDictionary_copy[field].replace(np.nan, fixValueOutput)
                    if missing_values is not None:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: fixValueOutput if x in missing_values else x)
                elif specialTypeInput == SpecialType.INVALID:
                    if missing_values is not None:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: fixValueOutput if x in missing_values else x)
                elif specialTypeInput == SpecialType.OUTLIER:
                    if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                        threshold = 1.5
                        Q1 = dataDictionary_copy[field].quantile(0.25)
                        Q3 = dataDictionary_copy[field].quantile(0.75)
                        IQR = Q3 - Q1
                        dataDictionary_copy[field] = dataDictionary_copy[field].where(
                            ~((dataDictionary_copy[field] < Q1 - threshold * IQR) |
                              (dataDictionary_copy[field] > Q3 + threshold * IQR)),
                            other=fixValueOutput)
                    else:
                        raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def transform_SpecialValue_DerivedValue(self, dataDictionary: pd.DataFrame, specialTypeInput: SpecialType,
                                            derivedTypeOutput: DerivedType, missing_values: list = None,
                                            axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the SpecialValue - DerivedValue relation
        :param dataDictionary: dataframe with the data
        :param specialTypeInput: special type of the input value
        :param derivedTypeOutput: derived type of the output value
        :param missing_values: list of missing values
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation
        :return: dataDictionary with the values of the special type changed to the value derived from the operation derivedTypeOutput
        """
        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
                dataDictionary_copy = apply_derivedType(specialTypeInput, derivedTypeOutput, dataDictionary_copy,
                                                        missing_values, axis_param, field)

            elif specialTypeInput == SpecialType.OUTLIER:
                # IMPORTANT: The function getOutliers() does the same as apply_derivedTypeOutliers() but at the dataframe level.
                # If the outliers are applied at the dataframe level, previous and next cannot be applied.

                if axis_param is None:
                    dataDictionary_copy_copy = getOutliers(dataDictionary_copy, field, axis_param)
                    missing_values = dataDictionary_copy.where(dataDictionary_copy_copy == 1).stack().tolist()
                    dataDictionary_copy = apply_derivedType(specialTypeInput, derivedTypeOutput, dataDictionary_copy,
                                                            missing_values, axis_param, field)
                elif axis_param == 0 or axis_param == 1:
                    dataDictionary_copy_copy = getOutliers(dataDictionary_copy, field, axis_param)
                    dataDictionary_copy = apply_derivedTypeColRowOutliers(derivedTypeOutput, dataDictionary_copy,
                                                                          dataDictionary_copy_copy, axis_param, field)

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")
            elif field in dataDictionary.columns:
                if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                    if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
                        dataDictionary_copy = apply_derivedType(specialTypeInput, derivedTypeOutput, dataDictionary_copy,
                                                                missing_values, axis_param, field)
                    elif specialTypeInput == SpecialType.OUTLIER:
                        dataDictionary_copy_copy = getOutliers(dataDictionary_copy, field, axis_param)
                        dataDictionary_copy = apply_derivedTypeColRowOutliers(derivedTypeOutput, dataDictionary_copy,
                                                                              dataDictionary_copy_copy, axis_param, field)
                else:
                    raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def transform_SpecialValue_NumOp(self, dataDictionary: pd.DataFrame, specialTypeInput: SpecialType,
                                     numOpOutput: Operation,
                                     missing_values: list = None, axis_param: int = None,
                                     field: str = None) -> pd.DataFrame:
        """
        Execute the data transformation of the SpecialValue - NumOp relation
        :param dataDictionary: dataframe with the data
        :param specialTypeInput: special type of the input value
        :param numOpOutput: operation to execute the data transformation
        :param axis_param: axis to execute the data transformation
        :param field: field to execute the data transformation
        :return: dataDictionary with the values of the special type changed to the result of the operation numOpOutput
        """
        dataDictionary_copy = dataDictionary.copy()
        dataDictionary_copy_mask = None

        if specialTypeInput == SpecialType.OUTLIER:
            dataDictionary_copy_mask = getOutliers(dataDictionary_copy, field, axis_param)

        if numOpOutput == Operation.INTERPOLATION:
            dataDictionary_copy = specialTypeInterpolation(dataDictionary_copy=dataDictionary_copy,
                                                           specialTypeInput=specialTypeInput,
                                                           dataDictionary_copy_mask=dataDictionary_copy_mask,
                                                           missing_values=missing_values, axis_param=axis_param,
                                                           field=field)
        elif numOpOutput == Operation.MEAN:
            dataDictionary_copy = specialTypeMean(dataDictionary_copy=dataDictionary_copy,
                                                  specialTypeInput=specialTypeInput,
                                                  dataDictionary_copy_mask=dataDictionary_copy_mask,
                                                  missing_values=missing_values, axis_param=axis_param, field=field)
        elif numOpOutput == Operation.MEDIAN:
            dataDictionary_copy = specialTypeMedian(dataDictionary_copy=dataDictionary_copy,
                                                    specialTypeInput=specialTypeInput,
                                                    dataDictionary_copy_mask=dataDictionary_copy_mask,
                                                    missing_values=missing_values, axis_param=axis_param, field=field)
        elif numOpOutput == Operation.CLOSEST:
            dataDictionary_copy = specialTypeClosest(dataDictionary_copy=dataDictionary_copy,
                                                     specialTypeInput=specialTypeInput,
                                                     dataDictionary_copy_mask=dataDictionary_copy_mask,
                                                     missing_values=missing_values, axis_param=axis_param, field=field)

        return dataDictionary_copy
