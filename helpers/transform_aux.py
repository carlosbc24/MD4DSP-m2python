# Importing enumerations from packages
# Importing libraries
import math

import numpy as np
import pandas as pd

from helpers.auxiliar import find_closest_value, outlier_closest, truncate
from helpers.enumerations import DerivedType, SpecialType


def get_outliers(data_dictionary: pd.DataFrame, field: str = None, axis_param: int = None) -> pd.DataFrame:
    """
    Get the outliers of a dataframe. The Outliers are calculated using the iqr method,
    so the outliers are the values that are below q1 - 1.5 * iqr or above q3 + 1.5 * iqr
    :param data_dictionary: dataframe with the data
    :param field: field to get the outliers. If field is None, the outliers are calculated for the whole dataframe.
    :param axis_param: axis to get the outliers. If axis_param is None, the outliers
    are calculated for the whole dataframe.
    If axis_param is 0, the outliers are calculated for each column. If axis_param is 1,
    the outliers are calculated for each row.

    :return: dataframe with the outliers. The value 1 indicates that the value is an outlier
    and the value 0 indicates that the value is not an outlier

    """
    # Filter the dataframe to get only the numeric values
    data_dictionary_numeric = data_dictionary.select_dtypes(include=[np.number])

    data_dictionary_copy = data_dictionary.copy()
    # Inicialize the dataframe with the same index and columns as the original dataframe
    data_dictionary_copy.loc[:, :] = 0

    threshold = 1.5
    if field is None:
        if axis_param is None:
            q1 = data_dictionary_numeric.stack().quantile(0.25)
            q3 = data_dictionary_numeric.stack().quantile(0.75)
            iqr = q3 - q1
            # Define the limits to identify outliers
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            # Sets the value 1 in the dataframe data_dictionary_copy for
            # the outliers and the value 0 for the non-outliers
            for col in data_dictionary_numeric.columns:
                for idx, value in data_dictionary[col].items():
                    if value < lower_bound or value > upper_bound:
                        data_dictionary_copy.at[idx, col] = 1
            return data_dictionary_copy

        elif axis_param == 0:
            for col in data_dictionary_numeric.columns:
                q1 = data_dictionary_numeric[col].quantile(0.25)
                q3 = data_dictionary_numeric[col].quantile(0.75)
                iqr = q3 - q1
                # Define the limits to identify outliers
                lower_bound_col = q1 - threshold * iqr
                upper_bound_col = q3 + threshold * iqr

                for idx, value in data_dictionary[col].items():
                    if value < lower_bound_col or value > upper_bound_col:
                        data_dictionary_copy.at[idx, col] = 1
            return data_dictionary_copy

        elif axis_param == 1:
            for idx, row in data_dictionary_numeric.iterrows():
                q1 = row.quantile(0.25)
                q3 = row.quantile(0.75)
                iqr = q3 - q1
                # Define the limits to identify outliers
                lower_bound_row = q1 - threshold * iqr
                upper_bound_row = q3 + threshold * iqr

                for col in row.index:
                    value = row[col]
                    if value < lower_bound_row or value > upper_bound_row:
                        data_dictionary_copy.at[idx, col] = 1
            return data_dictionary_copy
    elif field is not None:
        if not np.issubdtype(data_dictionary[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        q1 = data_dictionary[field].quantile(0.25)
        q3 = data_dictionary[field].quantile(0.75)
        iqr = q3 - q1

        lower_bound_col = q1 - threshold * iqr
        upper_bound_col = q3 + threshold * iqr

        for idx, value in data_dictionary[field].items():
            if value < lower_bound_col or value > upper_bound_col:
                data_dictionary_copy.at[idx, field] = 1

        return data_dictionary_copy


def apply_derived_type_col_row_outliers(derived_type_output: DerivedType, data_dictionary_copy: pd.DataFrame,
                                        data_dictionary_copy_copy: pd.DataFrame,
                                        axis_param: int = None, field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Apply the derived type to the outliers of a dataframe
    :param derived_type_output: derived type to apply to the outliers
    :param data_dictionary_copy: dataframe with the data
    :param data_dictionary_copy_copy: dataframe with the outliers
    :param axis_param: axis to apply the derived type. If axis_param is None,
    the derived type is applied to the whole dataframe.
    If axis_param is 0, the derived type is applied to each column. If axis_param is 1,
    the derived type is applied to each row.
    :param field_in: field to apply the derived type.
    :param field_out: field to store the result of the derived type.

    :return: dataframe with the derived type applied to the outliers
    """
    if field_in is None:
        if derived_type_output == DerivedType.MOSTFREQUENT:
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        for idx, value in data_dictionary_copy[col].items():
                            if data_dictionary_copy_copy.at[idx, col] == 1:
                                data_dictionary_copy.at[idx, col] = data_dictionary_copy[col].value_counts().idxmax()
            elif axis_param == 1:
                for idx, row in data_dictionary_copy.iterrows():
                    for col in row.index:
                        if data_dictionary_copy_copy.at[idx, col] == 1:
                            data_dictionary_copy.at[idx, col] = data_dictionary_copy.loc[idx].value_counts().idxmax()

        elif derived_type_output == DerivedType.PREVIOUS:
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    for idx, value in data_dictionary_copy[col].items():
                        if data_dictionary_copy_copy.at[idx, col] == 1 and idx != 0:
                            data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx - 1, col]
            elif axis_param == 1:
                for idx, row in data_dictionary_copy.iterrows():
                    for col in row.index:
                        if data_dictionary_copy_copy.at[idx, col] == 1 and col != 0:
                            prev_col = row.index[row.index.get_loc(col) - 1]  # Get the previous column
                            data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, prev_col]

        elif derived_type_output == DerivedType.NEXT:
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    for idx, value in data_dictionary_copy[col].items():
                        if data_dictionary_copy_copy.at[idx, col] == 1 and idx != len(data_dictionary_copy) - 1:
                            data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx + 1, col]
            elif axis_param == 1:
                for col in data_dictionary_copy.columns:
                    for idx, value in data_dictionary_copy[col].items():
                        if data_dictionary_copy_copy.at[idx, col] == 1 and col != data_dictionary_copy.columns[-1]:
                            next_col = data_dictionary_copy.columns[data_dictionary_copy.columns.get_loc(col) + 1]
                            data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, next_col]

    elif field_in is not None:
        if field_in not in data_dictionary_copy.columns or field_out not in data_dictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")

        if derived_type_output == DerivedType.MOSTFREQUENT:
            for idx, value in data_dictionary_copy[field_in].items():
                if data_dictionary_copy_copy.at[idx, field_in] == 1:
                    data_dictionary_copy.at[idx, field_out] = data_dictionary_copy[field_in].value_counts().idxmax()
                else:
                    data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_in]
        elif derived_type_output == DerivedType.PREVIOUS:
            for idx, value in data_dictionary_copy[field_in].items():
                if data_dictionary_copy_copy.at[idx, field_in] == 1 and idx != 0:
                    data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx - 1, field_in]
                else:
                    data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_in]
        elif derived_type_output == DerivedType.NEXT:
            for idx, value in data_dictionary_copy[field_in].items():
                if data_dictionary_copy_copy.at[idx, field_in] == 1 and idx != len(data_dictionary_copy) - 1:
                    data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx + 1, field_in]
                else:
                    data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_in]

    return data_dictionary_copy


def apply_derived_type(special_type_input: SpecialType, derived_type_output: DerivedType,
                       data_dictionary_copy: pd.DataFrame, missing_values: list = None, axis_param: int = None,
                       field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Apply the derived type to the missing values of a dataframe
    :param special_type_input: special type to apply to the missing values
    :param derived_type_output: derived type to apply to the missing values
    :param data_dictionary_copy: dataframe with the data
    :param missing_values: list of missing values
    :param axis_param: axis to apply the derived type. If axis_param is None,
    the derived type is applied to the whole dataframe.
    If axis_param is 0, the derived type is applied to each column. If axis_param is 1,
    the derived type is applied to each row.
    :param field_in: field to apply the derived type.
    :param field_out: field to store the result of the derived type.

    :return: dataframe with the derived type applied to the missing values
    """

    if field_in is None:
        if derived_type_output == DerivedType.MOSTFREQUENT:
            if axis_param == 0:
                if special_type_input == SpecialType.MISSING:
                    for col in data_dictionary_copy.columns: # Only missing
                        most_frequent = data_dictionary_copy[col].value_counts().idxmax()
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(lambda x: most_frequent if pd.isnull(x) else x)
                if missing_values is not None: # It works for missing values, invalid values and outliers
                    for col in data_dictionary_copy.columns:
                        most_frequent = data_dictionary_copy[col].value_counts().idxmax()
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(lambda x: most_frequent if x in missing_values else x)
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T
                if special_type_input == SpecialType.MISSING:
                    for row in data_dictionary_copy.columns:
                        most_frequent = data_dictionary_copy[row].value_counts().idxmax()
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                            lambda x: most_frequent if pd.isnull(x) else x)
                if missing_values is not None: # It works for missing values, invalid values and outliers
                    for row in data_dictionary_copy.columns:
                        most_frequent = data_dictionary_copy[row].value_counts().idxmax()
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                            lambda x: most_frequent if x in missing_values else x)
                data_dictionary_copy = data_dictionary_copy.T
            elif axis_param is None:
                valor_mas_frecuente = data_dictionary_copy.stack().value_counts().idxmax()
                if missing_values is not None: # It works for missing values, invalid values and outliers
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda col: col.apply(lambda x: valor_mas_frecuente if x in missing_values else x))
                if special_type_input == SpecialType.MISSING:
                    data_dictionary_copy = data_dictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: data_dictionary_copy[col.name].value_counts().idxmax() if pd.isnull(x) else x))

        elif derived_type_output == DerivedType.PREVIOUS:
            # Applies the lambda function in a column level or row level to replace the values within missing values by the value of the previous position
            if axis_param == 0 or axis_param == 1:
                if special_type_input == SpecialType.MISSING:
                    data_dictionary_copy = data_dictionary_copy.apply(lambda row_or_col: pd.Series([row_or_col.iloc[i - 1]
                                          if (value in missing_values or pd.isnull(value)) and i > 0 else value
                                           for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else: # Define the lambda function to replace the values within missing values by the value of the previous position
                    # It works for invalid values and outliers
                    data_dictionary_copy = data_dictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i - 1] if value in missing_values and i > 0 else value
                              for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)

            elif axis_param is None:
                raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

        elif derived_type_output == DerivedType.NEXT:
            # Define the lambda function to replace the values within missing values by the value of the next position
            if axis_param == 0 or axis_param == 1:
                if special_type_input == SpecialType.MISSING:
                    data_dictionary_copy = data_dictionary_copy.apply(lambda row_or_col: pd.Series([row_or_col.iloc[i + 1]
                                             if (value in missing_values or pd.isnull(value)) and i < len(row_or_col) - 1
                                                else value for i, value in enumerate(row_or_col)], index=row_or_col.index),
                                                    axis=axis_param)
                else: # It works for invalid values and outliers
                    data_dictionary_copy = data_dictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                                            value) else row_or_col.iloc[i + 1] if value in missing_values and i < len(
                                            row_or_col) - 1 else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index), axis=axis_param)
            elif axis_param is None:
                raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field_in is not None:
        if field_in not in data_dictionary_copy.columns or field_out not in data_dictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")

        if derived_type_output == DerivedType.MOSTFREQUENT:
            most_frequent = data_dictionary_copy[field_in].value_counts().idxmax()
            if special_type_input == SpecialType.MISSING:
                data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                    lambda x: most_frequent if pd.isnull(x) else x)
            if missing_values is not None: # It works for missing values, invalid values and outliers
                data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                    lambda x: most_frequent if x in missing_values else x)
        elif derived_type_output == DerivedType.PREVIOUS:
            if special_type_input == SpecialType.MISSING:
                if missing_values is not None:
                    data_dictionary_copy[field_out] = pd.Series([data_dictionary_copy[field_in].iloc[i - 1]
                                                if (value in missing_values or pd.isnull(value)) and i > 0 else value
                                                    for i, value in enumerate(data_dictionary_copy[field_in])],
                                                        index=data_dictionary_copy[field_in].index)
                else:
                    data_dictionary_copy[field_out] = pd.Series([data_dictionary_copy[field_in].iloc[i - 1] if pd.isnull(value)
                                                and i > 0 else value for i, value in enumerate(data_dictionary_copy[field_in])],
                                                    index=data_dictionary_copy[field_in].index)
            elif special_type_input == SpecialType.INVALID:
                if missing_values is not None: # It works for missing values, invalid values and outliers
                    data_dictionary_copy[field_out] = pd.Series(
                        [np.nan if pd.isnull(value) else data_dictionary_copy[field_in].iloc[i - 1]
                        if value in missing_values and i > 0 else value for i, value in
                         enumerate(data_dictionary_copy[field_in])], index=data_dictionary_copy[field_in].index)

        elif derived_type_output == DerivedType.NEXT:
            if special_type_input == SpecialType.MISSING:
                if missing_values is not None:
                    data_dictionary_copy[field_out] = pd.Series([data_dictionary_copy[field_in].iloc[i + 1]
                                                            if (value in missing_values or pd.isnull(
                                    value)) and i < len(data_dictionary_copy[field_in]) - 1 else value for i, value in
                                                            enumerate(data_dictionary_copy[field_in])],
                                                           index=data_dictionary_copy[field_in].index)
                else:
                    data_dictionary_copy[field_out] = pd.Series([data_dictionary_copy[field_in].iloc[i + 1]
                                                            if pd.isnull(value) and i < len(
                                                            data_dictionary_copy[field_in]) - 1 else value for i, value in
                                                            enumerate(data_dictionary_copy[field_in])],
                                                           index=data_dictionary_copy[field_in].index)
            elif special_type_input == SpecialType.INVALID:
                if missing_values is not None:
                    data_dictionary_copy[field_out] = pd.Series(
                        [np.nan if pd.isnull(value) else data_dictionary_copy[field_in].iloc[i + 1]
                        if value in missing_values and i < len(data_dictionary_copy[field_in]) - 1 else value for
                         i, value in enumerate(data_dictionary_copy[field_in])], index=data_dictionary_copy[field_in].index)

    return data_dictionary_copy


def special_type_interpolation(data_dictionary_copy: pd.DataFrame, special_type_input: SpecialType,
                               data_dictionary_copy_mask: pd.DataFrame = None,
                               missing_values: list = None, axis_param: int = None,
                               field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Apply the interpolation to the missing values of a dataframe
    :param data_dictionary_copy: dataframe with the data
    :param special_type_input: special type to apply to the missing values
    :param missing_values: list of missing values
    :param axis_param: axis to apply the interpolation.
    :param field_in: field to apply the interpolation.
    :param field_out: field to store the result of the interpolation.

    :return: dataframe with the interpolation applied to the missing values
    """
    data_dictionary_copy_copy = data_dictionary_copy.copy()

    if field_in is None:
        if axis_param is None:
            raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

        if special_type_input == SpecialType.MISSING:
            # Applies the linear interpolation in the DataFrame
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(lambda x: np.nan if x in missing_values else x)
                        data_dictionary_copy[col]=data_dictionary_copy[col].interpolate(method='linear', limit_direction='both')
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)

            elif axis_param == 1:
                data_dictionary_copy= data_dictionary_copy.T
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(lambda x: np.nan if x in missing_values else x)
                        data_dictionary_copy[col]=data_dictionary_copy[col].interpolate(method='linear', limit_direction='both')
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
                data_dictionary_copy = data_dictionary_copy.T

        if special_type_input == SpecialType.INVALID:
            # Applies the linear interpolation in the DataFrame
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(lambda x: np.nan if x in missing_values else x)
                        data_dictionary_copy_copy[col]=data_dictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy_copy.index:
                                if data_dictionary_copy_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy_copy.at[idx, col] = math.ceil(data_dictionary_copy_copy.at[idx, col])
                                else:
                                    data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy_copy.at[idx, col].round(0)
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
                data_dictionary_copy= data_dictionary_copy.T
                data_dictionary_copy_copy = data_dictionary_copy_copy.T
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].apply(lambda x: np.nan if x in missing_values else x)
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy_copy.index:
                                if data_dictionary_copy_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy_copy.at[idx, col] = math.ceil(data_dictionary_copy_copy.at[idx, col])
                                else:
                                    data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy_copy.at[idx, col].round(0)
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

        if special_type_input == SpecialType.OUTLIER:
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        for idx, value in data_dictionary_copy[col].items():
                            if data_dictionary_copy_mask.at[idx, col] == 1:
                                data_dictionary_copy_copy.at[idx, col] = np.NaN
                        data_dictionary_copy_copy[col] = data_dictionary_copy_copy[col].interpolate(method='linear', limit_direction='both')
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy_copy.index:
                                if data_dictionary_copy_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy_copy.at[idx, col] = math.ceil(data_dictionary_copy_copy.at[idx, col])
                                else:
                                    data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy_copy.at[idx, col].round(0)
                for col in data_dictionary_copy.columns:
                    # For each índex in the column
                    for idx in data_dictionary_copy.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_copy.at[idx, col]):
                            # Replace the value with the corresponding one from data_dictionary_copy_copy
                            data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                return data_dictionary_copy_copy
            elif axis_param == 1:
                data_dictionary_copy_copy=data_dictionary_copy_copy.T
                data_dictionary_copy=data_dictionary_copy.T
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        for idx, value in data_dictionary_copy[col].items():
                            if data_dictionary_copy_mask.at[idx, col] == 1:
                                data_dictionary_copy.at[idx, col] = np.NaN
                        data_dictionary_copy[col] = data_dictionary_copy[col].interpolate(method='linear', limit_direction='both')
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
                for col in data_dictionary_copy.columns:
                    # For each índex in the column
                    for idx in data_dictionary_copy.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(data_dictionary_copy.at[idx, col]):
                            # Replace the value with the corresponding one from data_dictionary_copy_copy
                            data_dictionary_copy_copy.at[idx, col] = data_dictionary_copy.at[idx, col]
                data_dictionary_copy_copy = data_dictionary_copy_copy.T
                return data_dictionary_copy_copy

    elif field_in is not None:
        if field_in not in data_dictionary_copy.columns or field_out not in data_dictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(data_dictionary_copy[field_in].dtype, np.number):
            raise ValueError("The field is not numeric")

        if special_type_input == SpecialType.MISSING:
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(lambda x: np.nan if x in missing_values else x)
            data_dictionary_copy[field_out]=data_dictionary_copy[field_in].interpolate(method='linear', limit_direction='both')
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy.index:
                    if data_dictionary_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy.at[idx, field_out] = math.ceil(data_dictionary_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_out].round(0)

        if special_type_input == SpecialType.INVALID:
            data_dictionary_copy_copy[field_out] = data_dictionary_copy[field_in].apply(lambda x: np.nan if x in missing_values else x)
            data_dictionary_copy_copy[field_out] = data_dictionary_copy_copy[field_in].interpolate(method='linear', limit_direction='both')
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy_copy.index:
                    if data_dictionary_copy_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy_copy.at[idx, field_out] = math.ceil(data_dictionary_copy_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy_copy.at[idx, field_out] = data_dictionary_copy_copy.at[idx, field_out].round(0)

            # For each índex in the column
            for idx in data_dictionary_copy.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(data_dictionary_copy.at[idx, field_in]):
                    # Replace the value with the corresponding one from data_dictionary_copy_copy
                    data_dictionary_copy_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_in]
            return data_dictionary_copy_copy

        if special_type_input == SpecialType.OUTLIER:
            for idx, value in data_dictionary_copy[field_in].items():
                if data_dictionary_copy_mask.at[idx, field_in] == 1:
                    data_dictionary_copy_copy.at[idx, field_in] = np.NaN
                else:
                    data_dictionary_copy_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_in]

            data_dictionary_copy_copy[field_out] = data_dictionary_copy_copy[field_in].interpolate(method='linear', limit_direction='both')
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy_copy.index:
                    if data_dictionary_copy_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy_copy.at[idx, field_out] = math.ceil(data_dictionary_copy_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy_copy.at[idx, field_out] = data_dictionary_copy_copy.at[idx, field_out].round(0)
            # For each índex in the column
            for idx in data_dictionary_copy.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(data_dictionary_copy.at[idx, field_in]):
                    # Replace the value with the corresponding one from data_dictionary_copy_copy
                    data_dictionary_copy_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_in]
            return data_dictionary_copy_copy

    return data_dictionary_copy


def special_type_mean(data_dictionary_copy: pd.DataFrame, special_type_input: SpecialType,
                      data_dictionary_copy_mask: pd.DataFrame = None,
                      missing_values: list = None, axis_param: int = None,
                      field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Apply the mean to the missing values of a dataframe
    :param data_dictionary_copy: dataframe with the data
    :param special_type_input: special type to apply to the missing values
    :param data_dictionary_copy_mask: dataframe with the outliers
    :param missing_values: list of missing values
    :param axis_param: axis to apply the mean.
    :param field_in: field to apply the mean.
    :param field_out: field to store the result of the mean.

    :return: dataframe with the mean applied to the missing values
    """

    if field_in is None:
        if special_type_input == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                # Calculate the mean of these numeric columns
                mean_value = only_numbers_df.mean().mean()
                # Replace the missing values with the mean of the entire DataFrame using lambda
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda col: col.apply(lambda x: mean_value if (x in missing_values or pd.isnull(x)) else x))
            elif axis_param == 0:
                means = data_dictionary_copy.apply(
                    lambda col: col[col.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                        col.dtype, np.number) else None)
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                            lambda x: x if not (x in missing_values or pd.isnull(x)) else means[col])
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T
                means = data_dictionary_copy.apply(
                    lambda row: row[row.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                        row.dtype, np.number) else None)
                for row in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[row].dtype, np.number):
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                            lambda x: x if not (x in missing_values or pd.isnull(x)) else means[row])
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[row].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, row] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, row] = math.ceil(data_dictionary_copy.at[idx, row])
                                else:
                                    data_dictionary_copy.at[idx, row] = data_dictionary_copy.at[idx, row].round(0)
                data_dictionary_copy = data_dictionary_copy.T
        if special_type_input == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                # Calculate the mean of these numeric columns
                mean_value = only_numbers_df.mean().mean()
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda col: col.apply(lambda x: mean_value if (x in missing_values) else x))

            elif axis_param == 0:
                means = data_dictionary_copy.apply(lambda col: col[col.apply(lambda x:
                        np.issubdtype(type(x), np.number))].mean() if np.issubdtype(col.dtype, np.number) else None)
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                            lambda x: x if not (x in missing_values) else means[col])
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T
                means = data_dictionary_copy.apply(lambda row: row[row.apply(lambda x:
                        np.issubdtype(type(x), np.number))].mean() if np.issubdtype(row.dtype, np.number) else None)
                for row in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[row].dtype, np.number):
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                            lambda x: x if not (x in missing_values) else means[row])
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[row].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, row] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, row] = math.ceil(data_dictionary_copy.at[idx, row])
                                else:
                                    data_dictionary_copy.at[idx, row] = data_dictionary_copy.at[idx, row].round(0)
                data_dictionary_copy = data_dictionary_copy.T

        if special_type_input == SpecialType.OUTLIER:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                # Calculate the mean of these numeric columns
                mean_value = only_numbers_df.mean().mean()
                # Replace the missing values with the mean of the entire DataFrame using lambda
                for col_name in data_dictionary_copy.columns:
                    for idx, value in data_dictionary_copy[col_name].items():
                        if np.issubdtype(type(value), np.number) and data_dictionary_copy_mask.at[idx, col_name] == 1:
                            data_dictionary_copy.at[idx, col_name] = mean_value
            if axis_param == 0: # Iterate over each column
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        mean=data_dictionary_copy[col].mean()
                        for idx, value in data_dictionary_copy[col].items():
                            if data_dictionary_copy_mask.at[idx, col] == 1:
                                data_dictionary_copy.at[idx, col] = mean
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
            elif axis_param == 1: # Iterate over each row
                for idx, row in data_dictionary_copy.iterrows():
                    if np.issubdtype(data_dictionary_copy[row].dtype, np.number):
                        mean=data_dictionary_copy.loc[idx].mean()
                        for col in row.index:
                            if data_dictionary_copy_mask.at[idx, col] == 1:
                                data_dictionary_copy.at[idx, col] = mean
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[row].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, row] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, row] = math.ceil(data_dictionary_copy.at[idx, row])
                                else:
                                    data_dictionary_copy.at[idx, row] = data_dictionary_copy.at[idx, row].round(0)

    elif field_in is not None:
        if field_in not in data_dictionary_copy.columns or field_out not in data_dictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(data_dictionary_copy[field_in].dtype, np.number):
            raise ValueError("The field is not numeric")

        if special_type_input == SpecialType.MISSING:
            mean = data_dictionary_copy[field_in].mean()
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                lambda x: mean if (x in missing_values or pd.isnull(x)) else x)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy.index:
                    if data_dictionary_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy.at[idx, field_out] = math.ceil(data_dictionary_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_out].round(0)

        if special_type_input == SpecialType.INVALID:
            mean = data_dictionary_copy[field_in].mean()
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                lambda x: mean if x in missing_values else x)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy.index:
                    if data_dictionary_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy.at[idx, field_out] = math.ceil(data_dictionary_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_out].round(0)

        if special_type_input == SpecialType.OUTLIER:
            mean=data_dictionary_copy[field_in].mean()
            for idx, value in data_dictionary_copy[field_in].items():
                if data_dictionary_copy_mask.at[idx, field_in] == 1:
                    data_dictionary_copy.at[idx, field_out] = mean
                else:
                    data_dictionary_copy.at[idx, field_out] = value
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy.index:
                    if data_dictionary_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy.at[idx, field_out] = math.ceil(data_dictionary_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_out].round(0)


    return data_dictionary_copy


def special_type_median(data_dictionary_copy: pd.DataFrame, special_type_input: SpecialType,
                        data_dictionary_copy_mask: pd.DataFrame = None,
                        missing_values: list = None, axis_param: int = None,
                        field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Apply the median to the missing values of a dataframe
    :param data_dictionary_copy: dataframe with the data
    :param special_type_input: special type to apply to the missing values
    :param missing_values: list of missing values
    :param axis_param: axis to apply the median.
    :param field_in: field to apply the median.
    :param field_out: field to store the result of the median.

    :return: dataframe with the median applied to the missing values
    """
    if field_in is None:
        if special_type_input == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda col: col.apply(lambda x: median_value if (x in missing_values or pd.isnull(x)) else x))
            elif axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        # Calculate the median without taking into account the nan values
                        median = data_dictionary_copy[col].median()
                        # Replace the values in missing_values, as well as the null values of python, by the median
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                            lambda x: median if x in missing_values or pd.isnull(x) else x)
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T
                medians = data_dictionary_copy.apply(
                    lambda row: row[row.apply(lambda x: np.issubdtype(type(x), np.number))].median() if np.issubdtype(
                        row.dtype, np.number) else None)
                for row in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[row].dtype, np.number):
                        data_dictionary_copy[row] = data_dictionary_copy[row].apply(
                            lambda x: x if not (x in missing_values or pd.isnull(x)) else medians[row])
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[row].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, row] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, row] = math.ceil(data_dictionary_copy.at[idx, row])
                                else:
                                    data_dictionary_copy.at[idx, row] = data_dictionary_copy.at[idx, row].round(0)
                data_dictionary_copy = data_dictionary_copy.T

        if special_type_input == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Replace the values in missing_values (INVALID VALUES) by the median
                data_dictionary_copy = data_dictionary_copy.apply(
                    lambda col: col.apply(lambda x: median_value if (x in missing_values) else x))
            elif axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        # Calculate the median without taking into account the nan values
                        median = data_dictionary_copy[col].median()
                        # Replace the values in missing_values, as well as the null values of python, by the median
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                            lambda x: median if x in missing_values else x)
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
            elif axis_param == 1:
                data_dictionary_copy = data_dictionary_copy.T
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        # Calculate the median without taking into account the nan values
                        median = data_dictionary_copy[col].median()
                        # Replace the values in missing_values, as well as the null values of python, by the median
                        data_dictionary_copy[col] = data_dictionary_copy[col].apply(
                            lambda x: median if x in missing_values else x)
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
                data_dictionary_copy = data_dictionary_copy.T

        if special_type_input == SpecialType.OUTLIER:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Replace the outliers with the median of the entire DataFrame using lambda
                for col_name in data_dictionary_copy.columns:
                    for idx, value in data_dictionary_copy[col_name].items():
                        if data_dictionary_copy_mask.at[idx, col_name] == 1:
                            data_dictionary_copy.at[idx, col_name] = median_value
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    if (data_dictionary_copy[col_name].dropna() % 1 == 0).all():
                        for idx in data_dictionary_copy.index:
                            if data_dictionary_copy.at[idx, col_name] % 1 >= 0.5:
                                data_dictionary_copy.at[idx, col_name] = math.ceil(data_dictionary_copy.at[idx, col_name])
                            else:
                                data_dictionary_copy.at[idx, col_name] = data_dictionary_copy.at[idx, col_name].round(0)
            if axis_param == 0:
                for col in data_dictionary_copy.columns:
                    if np.issubdtype(data_dictionary_copy[col].dtype, np.number):
                        median=data_dictionary_copy[col].median()
                        for idx, value in data_dictionary_copy[col].items():
                            if data_dictionary_copy_mask.at[idx, col] == 1:
                                data_dictionary_copy.at[idx, col] = median
                        # Trunk the decimals to 0 if the column is int or if it has no decimals
                        if (data_dictionary_copy[col].dropna() % 1 == 0).all():
                            for idx in data_dictionary_copy.index:
                                if data_dictionary_copy.at[idx, col] % 1 >= 0.5:
                                    data_dictionary_copy.at[idx, col] = math.ceil(data_dictionary_copy.at[idx, col])
                                else:
                                    data_dictionary_copy.at[idx, col] = data_dictionary_copy.at[idx, col].round(0)
            elif axis_param == 1:
                for idx, row in data_dictionary_copy.iterrows():
                    median=data_dictionary_copy.loc[idx].median()
                    for col in row.index:
                        if data_dictionary_copy_mask.at[idx, col] == 1:
                            data_dictionary_copy.at[idx, col] = median
                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    if (data_dictionary_copy[row].dropna() % 1 == 0).all():
                        for idx in data_dictionary_copy.index:
                            if data_dictionary_copy.loc[idx] % 1 >= 0.5:
                                data_dictionary_copy.loc[idx] = math.ceil(data_dictionary_copy.loc[idx])
                            else:
                                data_dictionary_copy.loc[idx] = data_dictionary_copy.loc[idx].round(0)
    elif field_in is not None:
        if field_in not in data_dictionary_copy.columns or field_out not in data_dictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(data_dictionary_copy[field_in].dtype, np.number):
            raise ValueError("The field is not numeric")

        if special_type_input == SpecialType.MISSING:
            median = data_dictionary_copy[field_in].median()
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                lambda x: median if x in missing_values or pd.isnull(x) else x)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy.index:
                    if data_dictionary_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy.at[idx, field_out] = math.ceil(data_dictionary_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_out].round(0)

        if special_type_input == SpecialType.INVALID:
            median = data_dictionary_copy[field_in].median()
            data_dictionary_copy[field_out] = data_dictionary_copy[field_in].apply(
                lambda x: median if x in missing_values else x)
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy.index:
                    if data_dictionary_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy.at[idx, field_out] = math.ceil(data_dictionary_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_out].round(0)

        if special_type_input == SpecialType.OUTLIER:
            median = data_dictionary_copy[field_in].median()
            for idx, value in data_dictionary_copy[field_in].items():
                if data_dictionary_copy_mask.at[idx, field_in] == 1:
                    data_dictionary_copy.at[idx, field_out] = median
                else:
                    data_dictionary_copy.at[idx, field_out] = value
            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy.index:
                    if data_dictionary_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy.at[idx, field_out] = math.ceil(data_dictionary_copy.at[idx, field_out])
                    else:
                        data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_out].round(0)

    return data_dictionary_copy


def special_type_closest(data_dictionary_copy: pd.DataFrame, special_type_input: SpecialType,
                         data_dictionary_copy_mask: pd.DataFrame = None,
                         missing_values: list = None, axis_param: int = None,
                         field_in: str = None, field_out: str = None) -> pd.DataFrame:
    """
    Apply the closest to the missing values of a dataframe
    :param data_dictionary_copy: dataframe with the data
    :param special_type_input: special type to apply to the missing values
    :param data_dictionary_copy_mask: dataframe with the outliers mask
    :param missing_values: list of missing values
    :param axis_param: axis to apply the closest value.
    :param field_in: field to apply the closest value.
    :param field_out: field to store the closest value.

    :return: dataframe with the closest applied to the missing values
    """

    if field_in is None:
        if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
            if axis_param is None:
                only_numbers_df = data_dictionary_copy.select_dtypes(include=[np.number])
                # Flatten the DataFrame into a single series of values
                flattened_values = only_numbers_df.values.flatten().tolist()
                # Create a dictionary to store the closest value for each missing value
                closest_values = {}
                # For each missing value, find the closest numeric value in the flattened series
                for missing_value in missing_values:
                    if missing_value not in closest_values:
                        closest_values[missing_value] = find_closest_value(flattened_values, missing_value)
                # Replace the missing values with the closest numeric values
                for i in range(len(data_dictionary_copy.index)):
                    for col, value in data_dictionary_copy.iloc[i].items():
                        current_value = data_dictionary_copy.at[i, col]
                        if current_value in closest_values:
                            data_dictionary_copy.at[i, col] = closest_values[current_value]
                        else:
                            if pd.isnull(data_dictionary_copy.at[i, col]) and special_type_input == SpecialType.MISSING:
                                raise ValueError(
                                    "Error: it's not possible to apply the closest operation to the null values")

            elif axis_param == 0:
                # Iterate over each column
                for col_name in data_dictionary_copy.select_dtypes(include=[np.number]).columns:
                    # Get the missing values in the current column
                    missing_values_in_col = [val for val in missing_values if val in data_dictionary_copy[col_name].values]
                    # If there are no missing values in the column, skip the rest of the loop
                    if not missing_values_in_col:
                        continue
                    # Flatten the column into a list of values
                    flattened_values = data_dictionary_copy[col_name].values.flatten().tolist()
                    # Create a dictionary to store the closest value for each missing value
                    closest_values = {}
                    # For each missing value IN the column (more efficient), find the closest numeric value in the flattened series
                    for missing_value in missing_values_in_col:
                        if missing_value not in closest_values:
                            closest_values[missing_value] = find_closest_value(flattened_values, missing_value)
                    # Replace the missing values with the closest numeric values in the column
                    for i in range(len(data_dictionary_copy.index)):
                        current_value = data_dictionary_copy.at[i, col_name]
                        if current_value in closest_values:
                            data_dictionary_copy.at[i, col_name] = closest_values[current_value]
                        else:
                            if pd.isnull(data_dictionary_copy.at[i, col_name]) and special_type_input == SpecialType.MISSING:
                                raise ValueError(
                                    "Error: it's not possible to apply the closest operation to the null values")
            elif axis_param == 1:
                # Iterate over each row
                for row_idx in range(len(data_dictionary_copy.index)):
                    # Get the numeric values in the current row
                    numeric_values_in_row = data_dictionary_copy.iloc[row_idx].select_dtypes(
                        include=[np.number]).values.tolist()
                    # Get the missing values in the current row
                    missing_values_in_row = [val for val in missing_values if val in numeric_values_in_row]
                    # If there are no missing values in the row, skip the rest of the loop
                    if not missing_values_in_row and not pd.isnull(data_dictionary_copy.iloc[row_idx]).any():
                        continue
                    # Create a dictionary to store the closest value for each missing value
                    closest_values = {}
                    # For each missing value IN the row (more efficient), find the closest numeric value in the numeric values
                    for missing_value in missing_values_in_row:
                        if missing_value not in closest_values:
                            closest_values[missing_value] = find_closest_value(numeric_values_in_row, missing_value)
                    # Replace the missing values with the closest numeric values in the row
                    for col_name in data_dictionary_copy.columns:
                        current_value = data_dictionary_copy.at[row_idx, col_name]
                        if current_value in closest_values:
                            data_dictionary_copy.at[row_idx, col_name] = closest_values[current_value]
                        else:
                            if pd.isnull(data_dictionary_copy.at[row_idx, col_name]) and special_type_input == SpecialType.MISSING:
                                raise ValueError("Error: it's not possible to apply the closest operation to the null values")

        if special_type_input == SpecialType.OUTLIER:
            if axis_param is None:
                minimum_valid, maximum_valid=outlier_closest(data_dictionary=data_dictionary_copy,
                                                             axis_param=None, field=None)

                # Replace the outlier values with the closest numeric values
                for i in range(len(data_dictionary_copy.index)):
                    for j in range(len(data_dictionary_copy.columns)):
                        if data_dictionary_copy_mask.at[i, j] == 1:
                            if data_dictionary_copy.at[i, j] > maximum_valid:
                                data_dictionary_copy.at[i, j] = maximum_valid
                            elif data_dictionary_copy.at[i, j] < minimum_valid:
                                data_dictionary_copy.at[i, j] = minimum_valid
            elif axis_param == 0:
                # Iterate over each column
                for col_name in data_dictionary_copy.select_dtypes(include=[np.number]).columns:
                    minimum_valid, maximum_valid = outlier_closest(data_dictionary=data_dictionary_copy,
                                                                   axis_param=0, field=col_name)

                    # Trunk the decimals to 0 if the column is int or if it has no decimals
                    if (data_dictionary_copy[col_name].dropna() % 1 == 0).all():
                        for idx in data_dictionary_copy.index:
                            if data_dictionary_copy.at[idx, col_name] % 1 >= 0.5:
                                data_dictionary_copy.at[idx, col_name] = math.ceil(data_dictionary_copy.at[idx, col_name])
                                minimum_valid = math.ceil(minimum_valid)
                                maximum_valid = math.ceil(maximum_valid)
                            else:
                                data_dictionary_copy.at[idx, col_name] = data_dictionary_copy.at[idx, col_name].round(0)
                                minimum_valid = minimum_valid.round(0)
                                maximum_valid = maximum_valid.round(0)

                    # Replace the outlier values with the closest numeric values
                    for i in range(len(data_dictionary_copy.index)):
                        if data_dictionary_copy_mask.at[i, col_name] == 1:
                            if data_dictionary_copy.at[i, col_name] > maximum_valid:
                                data_dictionary_copy.at[i, col_name] = maximum_valid
                            elif data_dictionary_copy.at[i, col_name] < minimum_valid:
                                data_dictionary_copy.at[i, col_name] = minimum_valid

    elif field_in is not None:
        if field_in not in data_dictionary_copy.columns or field_out not in data_dictionary_copy.columns:
            raise ValueError("Field not found in the data_dictionary_in")
        if not np.issubdtype(data_dictionary_copy[field_in].dtype, np.number):
            raise ValueError("Field is not numeric")

        if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
            # Get the missing values in the current column
            missing_values_in_col = [val for val in missing_values if val in data_dictionary_copy[field_in].values]
            # If there are no missing values in the column, skip the rest of the loop
            if missing_values_in_col or pd.isnull(data_dictionary_copy[field_in]).any():
                # Flatten the column into a list of values
                flattened_values = data_dictionary_copy[field_in].values.flatten().tolist()
                # Create a dictionary to store the closest value for each missing value
                closest_values = {}
                # For each missing value IN the column (more efficient), find the closest numeric value in the flattened series
                for missing_value in missing_values_in_col:
                    if missing_value not in closest_values:
                        closest_values[missing_value] = find_closest_value(flattened_values, missing_value)
                # Replace the missing values with the closest numeric values in the column
                for i in range(len(data_dictionary_copy.index)):
                    current_value = data_dictionary_copy.at[i, field_in]
                    if current_value in closest_values:
                        data_dictionary_copy.at[i, field_out] = closest_values[current_value]
                    else:
                        if pd.isnull(data_dictionary_copy.at[i, field_in]) and special_type_input == SpecialType.MISSING:
                            raise ValueError("Error: it's not possible to apply the closest operation to the null values")

        if special_type_input == SpecialType.OUTLIER:
            minimum_valid, maximum_valid = outlier_closest(data_dictionary=data_dictionary_copy,
                                                           axis_param=None, field=field_in)

            # Trunk the decimals to 0 if the column is int or if it has no decimals
            if (data_dictionary_copy[field_out].dropna() % 1 == 0).all():
                for idx in data_dictionary_copy.index:
                    if data_dictionary_copy.at[idx, field_out] % 1 >= 0.5:
                        data_dictionary_copy.at[idx, field_out] = math.ceil(data_dictionary_copy.at[idx, field_out])
                        minimum_valid = math.ceil(minimum_valid)
                        maximum_valid = math.ceil(maximum_valid)
                    else:
                        data_dictionary_copy.at[idx, field_out] = data_dictionary_copy.at[idx, field_out].round(0)
                        minimum_valid = minimum_valid.round(0)
                        maximum_valid = maximum_valid.round(0)

            # Replace the outlier values with the closest numeric values
            for i in range(len(data_dictionary_copy.index)):
                if data_dictionary_copy_mask.at[i, field_in] == 1:
                    if data_dictionary_copy.at[i, field_in] > maximum_valid:
                        data_dictionary_copy.at[i, field_out] = maximum_valid
                    elif data_dictionary_copy.at[i, field_in] < minimum_valid:
                        data_dictionary_copy.at[i, field_out] = minimum_valid

    return data_dictionary_copy
