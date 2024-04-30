# Importing enumerations from packages
from helpers.auxiliar import find_closest_value
from helpers.enumerations import DerivedType, SpecialType

# Importing libraries
import numpy as np
import pandas as pd

def getOutliers(dataDictionary: pd.DataFrame, field: str = None, axis_param: int = None) -> pd.DataFrame:
    """
    Get the outliers of a dataframe. The Outliers are calculated using the IQR method, so the outliers are the values that are
    below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR
    :param dataDictionary: dataframe with the data
    :param field: field to get the outliers. If field is None, the outliers are calculated for the whole dataframe.
    :param axis_param: axis to get the outliers. If axis_param is None, the outliers are calculated for the whole dataframe.
    If axis_param is 0, the outliers are calculated for each column. If axis_param is 1, the outliers are calculated for each row.

    :return: dataframe with the outliers. The value 1 indicates that the value is an outlier and the value 0 indicates that the value is not an outlier

    """
    # Filter the dataframe to get only the numeric values
    dataDictionary_numeric = dataDictionary.select_dtypes(include=[np.number])

    dataDictionary_copy = dataDictionary.copy()
    # Inicialize the dataframe with the same index and columns as the original dataframe
    dataDictionary_copy.loc[:, :] = 0

    threshold = 1.5
    if field is None:
        if axis_param is None:
            Q1 = dataDictionary_numeric.stack().quantile(0.25)
            Q3 = dataDictionary_numeric.stack().quantile(0.75)
            IQR = Q3 - Q1
            # Define the limits to identify outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            # Sets the value 1 in the dataframe dataDictionary_copy for the outliers and the value 0 for the non-outliers
            for col in dataDictionary_numeric.columns:
                for idx, value in dataDictionary[col].items():
                    if value < lower_bound or value > upper_bound:
                        dataDictionary_copy.at[idx, col] = 1
            return dataDictionary_copy

        elif axis_param == 0:
            for col in dataDictionary_numeric.columns:
                Q1 = dataDictionary_numeric[col].quantile(0.25)
                Q3 = dataDictionary_numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                # Define the limits to identify outliers
                lower_bound_col = Q1 - threshold * IQR
                upper_bound_col = Q3 + threshold * IQR

                for idx, value in dataDictionary[col].items():
                    if value < lower_bound_col or value > upper_bound_col:
                        dataDictionary_copy.at[idx, col] = 1
            return dataDictionary_copy

        elif axis_param == 1:
            for idx, row in dataDictionary_numeric.iterrows():
                Q1 = row.quantile(0.25)
                Q3 = row.quantile(0.75)
                IQR = Q3 - Q1
                # Define the limits to identify outliers
                lower_bound_row = Q1 - threshold * IQR
                upper_bound_row = Q3 + threshold * IQR

                for col in row.index:
                    value = row[col]
                    if value < lower_bound_row or value > upper_bound_row:
                        dataDictionary_copy.at[idx, col] = 1
            return dataDictionary_copy
    elif field is not None:
        if not np.issubdtype(dataDictionary[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        Q1 = dataDictionary[field].quantile(0.25)
        Q3 = dataDictionary[field].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound_col = Q1 - threshold * IQR
        upper_bound_col = Q3 + threshold * IQR

        for idx, value in dataDictionary[field].items():
            if value < lower_bound_col or value > upper_bound_col:
                dataDictionary_copy.at[idx, field] = 1

        return dataDictionary_copy


def apply_derivedTypeColRowOutliers(derivedTypeOutput: DerivedType, dataDictionary_copy: pd.DataFrame,
                                    dataDictionary_copy_copy: pd.DataFrame,
                                    axis_param: int = None, field: str = None) -> pd.DataFrame:
    """
    Apply the derived type to the outliers of a dataframe
    :param derivedTypeOutput: derived type to apply to the outliers
    :param dataDictionary_copy: dataframe with the data
    :param dataDictionary_copy_copy: dataframe with the outliers
    :param axis_param: axis to apply the derived type. If axis_param is None, the derived type is applied to the whole dataframe.
    If axis_param is 0, the derived type is applied to each column. If axis_param is 1, the derived type is applied to each row.
    :param field: field to apply the derived type.

    :return: dataframe with the derived type applied to the outliers
    """
    if field is None:
        if derivedTypeOutput == DerivedType.MOSTFREQUENT:
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        for idx, value in dataDictionary_copy[col].items():
                            if dataDictionary_copy_copy.at[idx, col] == 1:
                                dataDictionary_copy.at[idx, col] = dataDictionary_copy[col].value_counts().idxmax()
            elif axis_param == 1:
                for idx, row in dataDictionary_copy.iterrows():
                    for col in row.index:
                        if dataDictionary_copy_copy.at[idx, col] == 1:
                            dataDictionary_copy.at[idx, col] = dataDictionary_copy.loc[idx].value_counts().idxmax()

        elif derivedTypeOutput == DerivedType.PREVIOUS:
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col].items():
                        if dataDictionary_copy_copy.at[idx, col] == 1 and idx != 0:
                            dataDictionary_copy.at[idx, col] = dataDictionary_copy.at[idx - 1, col]
            elif axis_param == 1:
                for idx, row in dataDictionary_copy.iterrows():
                    for col in row.index:
                        if dataDictionary_copy_copy.at[idx, col] == 1 and col != 0:
                            prev_col = row.index[row.index.get_loc(col) - 1]  # Get the previous column
                            dataDictionary_copy.at[idx, col] = dataDictionary_copy.at[idx, prev_col]

        elif derivedTypeOutput == DerivedType.NEXT:
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col].items():
                        if dataDictionary_copy_copy.at[idx, col] == 1 and idx != len(dataDictionary_copy) - 1:
                            dataDictionary_copy.at[idx, col] = dataDictionary_copy.at[idx + 1, col]
            elif axis_param == 1:
                for col in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col].items():
                        if dataDictionary_copy_copy.at[idx, col] == 1 and col != dataDictionary_copy.columns[-1]:
                            next_col = dataDictionary_copy.columns[dataDictionary_copy.columns.get_loc(col) + 1]
                            dataDictionary_copy.at[idx, col] = dataDictionary_copy.at[idx, next_col]

    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        elif field in dataDictionary_copy_copy.columns:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                for idx, value in dataDictionary_copy[field].items():
                    if dataDictionary_copy_copy.at[idx, field] == 1:
                        dataDictionary_copy.at[idx, field] = dataDictionary_copy[field].value_counts().idxmax()
            elif derivedTypeOutput == DerivedType.PREVIOUS:
                for idx, value in dataDictionary_copy[field].items():
                    if dataDictionary_copy_copy.at[idx, field] == 1 and idx != 0:
                        dataDictionary_copy.at[idx, field] = dataDictionary_copy.at[idx - 1, field]
            elif derivedTypeOutput == DerivedType.NEXT:
                for idx, value in dataDictionary_copy[field].items():
                    if dataDictionary_copy_copy.at[idx, field] == 1 and idx != len(dataDictionary_copy) - 1:
                        dataDictionary_copy.at[idx, field] = dataDictionary_copy.at[idx + 1, field]

    return dataDictionary_copy


def apply_derivedType(specialTypeInput: SpecialType, derivedTypeOutput: DerivedType, dataDictionary_copy: pd.DataFrame,
                      missing_values: list = None, axis_param: int = None, field: str = None) -> pd.DataFrame:
    """
    Apply the derived type to the missing values of a dataframe
    :param specialTypeInput: special type to apply to the missing values
    :param derivedTypeOutput: derived type to apply to the missing values
    :param dataDictionary_copy: dataframe with the data
    :param missing_values: list of missing values
    :param axis_param: axis to apply the derived type. If axis_param is None, the derived type is applied to the whole dataframe.
    If axis_param is 0, the derived type is applied to each column. If axis_param is 1, the derived type is applied to each row.
    :param field: field to apply the derived type.

    :return: dataframe with the derived type applied to the missing values
    """

    if field is None:
        if derivedTypeOutput == DerivedType.MOSTFREQUENT:
            if axis_param == 0:
                if specialTypeInput == SpecialType.MISSING:
                    for col in dataDictionary_copy.columns: # Only missing
                        most_frequent = dataDictionary_copy[col].value_counts().idxmax()
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(lambda x: most_frequent if pd.isnull(x) else x)
                if missing_values is not None: # It works for missing values, invalid values and outliers
                    for col in dataDictionary_copy.columns:
                        most_frequent = dataDictionary_copy[col].value_counts().idxmax()
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(lambda x: most_frequent if x in missing_values else x)
            elif axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.T
                if specialTypeInput == SpecialType.MISSING:
                    for row in dataDictionary_copy.columns:
                        most_frequent = dataDictionary_copy[row].value_counts().idxmax()
                        dataDictionary_copy[row] = dataDictionary_copy[row].apply(
                            lambda x: most_frequent if pd.isnull(x) else x)
                if missing_values is not None: # It works for missing values, invalid values and outliers
                    for row in dataDictionary_copy.columns:
                        most_frequent = dataDictionary_copy[row].value_counts().idxmax()
                        dataDictionary_copy[row] = dataDictionary_copy[row].apply(
                            lambda x: most_frequent if x in missing_values else x)
                dataDictionary_copy = dataDictionary_copy.T
            elif axis_param is None:
                valor_mas_frecuente = dataDictionary_copy.stack().value_counts().idxmax()
                if missing_values is not None: # It works for missing values, invalid values and outliers
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(lambda x: valor_mas_frecuente if x in missing_values else x))
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: dataDictionary_copy[col.name].value_counts().idxmax() if pd.isnull(x) else x))

        elif derivedTypeOutput == DerivedType.PREVIOUS:
            # Applies the lambda function in a column level or row level to replace the values within missing values by the value of the previous position
            if axis_param == 0 or axis_param == 1:
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([row_or_col.iloc[i - 1]
                                          if (value in missing_values or pd.isnull(value)) and i > 0 else value
                                           for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else: # Define the lambda function to replace the values within missing values by the value of the previous position
                    # It works for invalid values and outliers
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i - 1] if value in missing_values and i > 0 else value
                              for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)

            elif axis_param is None:
                raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

        elif derivedTypeOutput == DerivedType.NEXT:
            # Define the lambda function to replace the values within missing values by the value of the next position
            if axis_param == 0 or axis_param == 1:
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([row_or_col.iloc[i + 1]
                                             if (value in missing_values or pd.isnull(value)) and i < len(row_or_col) - 1
                                                else value for i, value in enumerate(row_or_col)], index=row_or_col.index),
                                                    axis=axis_param)
                else: # It works for invalid values and outliers
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                                            value) else row_or_col.iloc[i + 1] if value in missing_values and i < len(
                                            row_or_col) - 1 else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index), axis=axis_param)
            elif axis_param is None:
                raise ValueError("The axis cannot be None when applying the NEXT operation")

    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")

        elif field in dataDictionary_copy.columns:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                most_frequent = dataDictionary_copy[field].value_counts().idxmax()
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: most_frequent if pd.isnull(x) else x)
                if missing_values is not None: # It works for missing values, invalid values and outliers
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: most_frequent if x in missing_values else x)
            elif derivedTypeOutput == DerivedType.PREVIOUS:
                if specialTypeInput == SpecialType.MISSING:
                    if missing_values is not None:
                        dataDictionary_copy[field] = pd.Series([dataDictionary_copy[field].iloc[i - 1]
                                                    if (value in missing_values or pd.isnull(value)) and i > 0 else value
                                                        for i, value in enumerate(dataDictionary_copy[field])],
                                                            index=dataDictionary_copy[field].index)
                    else:
                        dataDictionary_copy[field] = pd.Series([dataDictionary_copy[field].iloc[i - 1] if pd.isnull(value)
                                                    and i > 0 else value for i, value in enumerate(dataDictionary_copy[field])],
                                                        index=dataDictionary_copy[field].index)
                elif specialTypeInput == SpecialType.INVALID:
                    if missing_values is not None: # It works for missing values, invalid values and outliers
                        dataDictionary_copy[field] = pd.Series(
                            [np.nan if pd.isnull(value) else dataDictionary_copy[field].iloc[i - 1]
                            if value in missing_values and i > 0 else value for i, value in
                             enumerate(dataDictionary_copy[field])], index=dataDictionary_copy[field].index)

            elif derivedTypeOutput == DerivedType.NEXT:
                if specialTypeInput == SpecialType.MISSING:
                    if missing_values is not None:
                        dataDictionary_copy[field] = pd.Series([dataDictionary_copy[field].iloc[i + 1]
                                                                if (value in missing_values or pd.isnull(
                            value)) and i < len(dataDictionary_copy[field]) - 1 else value for i, value in
                                                                enumerate(dataDictionary_copy[field])],
                                                               index=dataDictionary_copy[field].index)
                    else:
                        dataDictionary_copy[field] = pd.Series([dataDictionary_copy[field].iloc[i + 1]
                                                                if pd.isnull(value) and i < len(
                            dataDictionary_copy[field]) - 1 else value for i, value in
                                                                enumerate(dataDictionary_copy[field])],
                                                               index=dataDictionary_copy[field].index)
                elif specialTypeInput == SpecialType.INVALID:
                    if missing_values is not None:
                        dataDictionary_copy[field] = pd.Series(
                            [np.nan if pd.isnull(value) else dataDictionary_copy[field].iloc[i + 1]
                            if value in missing_values and i < len(dataDictionary_copy[field]) - 1 else value for
                             i, value in
                             enumerate(dataDictionary_copy[field])], index=dataDictionary_copy[field].index)

    return dataDictionary_copy


def specialTypeInterpolation(dataDictionary_copy: pd.DataFrame, specialTypeInput: SpecialType,
                             dataDictionary_copy_mask: pd.DataFrame = None,
                             missing_values: list = None, axis_param: int = None, field: str = None) -> pd.DataFrame:
    """
    Apply the interpolation to the missing values of a dataframe
    :param dataDictionary_copy: dataframe with the data
    :param specialTypeInput: special type to apply to the missing values
    :param missing_values: list of missing values
    :param axis_param: axis to apply the interpolation.
    :param field: field to apply the interpolation.

    :return: dataframe with the interpolation applied to the missing values
    """
    dataDictionary_copy_copy = dataDictionary_copy.copy()

    if field is None:
        if axis_param is None:
            raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

        if specialTypeInput == SpecialType.MISSING:
            # Applies the linear interpolation in the DataFrame
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(lambda x: np.nan if x in missing_values else x)
                        dataDictionary_copy[col]=dataDictionary_copy[col].interpolate(method='linear', limit_direction='both')

            elif axis_param == 1:
                dataDictionary_copy= dataDictionary_copy.T
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(lambda x: np.nan if x in missing_values else x)
                        dataDictionary_copy[col]=dataDictionary_copy[col].interpolate(method='linear', limit_direction='both')
                dataDictionary_copy = dataDictionary_copy.T


        if specialTypeInput == SpecialType.INVALID:
            # Applies the linear interpolation in the DataFrame
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].apply(lambda x: np.nan if x in missing_values else x)
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
                dataDictionary_copy= dataDictionary_copy.T
                dataDictionary_copy_copy = dataDictionary_copy_copy.T
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].apply(lambda x: np.nan if x in missing_values else x)
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

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        for idx, value in dataDictionary_copy[col].items():
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy_copy.at[idx, col] = np.NaN
                        dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].interpolate(method='linear',
                                                                                                limit_direction='both')
                for col in dataDictionary_copy.columns:
                    # For each índex in the column
                    for idx in dataDictionary_copy.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(dataDictionary_copy.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            dataDictionary_copy_copy.at[idx, col] = dataDictionary_copy.at[idx, col]
                return dataDictionary_copy_copy
            elif axis_param == 1:
                dataDictionary_copy_copy=dataDictionary_copy_copy.T
                dataDictionary_copy=dataDictionary_copy.T
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        for idx, value in dataDictionary_copy[col].items():
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy.at[idx, col] = np.NaN
                        dataDictionary_copy[col] = dataDictionary_copy[col].interpolate(method='linear',
                                                                                                limit_direction='both')
                for col in dataDictionary_copy.columns:
                    # For each índex in the column
                    for idx in dataDictionary_copy.index:
                        # Verify if the value is NaN in the original dataframe
                        if pd.isnull(dataDictionary_copy.at[idx, col]):
                            # Replace the value with the corresponding one from dataDictionary_copy_copy
                            dataDictionary_copy_copy.at[idx, col] = dataDictionary_copy.at[idx, col]
                dataDictionary_copy_copy = dataDictionary_copy_copy.T
                return dataDictionary_copy_copy
    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(dataDictionary_copy[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        if specialTypeInput == SpecialType.MISSING:
            dataDictionary_copy[field] = dataDictionary_copy[field].apply(lambda x: np.nan if x in missing_values else x)
            dataDictionary_copy[field]=dataDictionary_copy[field].interpolate(method='linear', limit_direction='both')

        if specialTypeInput == SpecialType.INVALID:
            dataDictionary_copy_copy[field] = dataDictionary_copy[field].apply(lambda x: np.nan if x in missing_values else x)
            dataDictionary_copy_copy[field] = dataDictionary_copy_copy[field].interpolate(method='linear', limit_direction='both')

            # For each índex in the column
            for idx in dataDictionary_copy.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(dataDictionary_copy.at[idx, field]):
                    # Replace the value with the corresponding one from dataDictionary_copy_copy
                    dataDictionary_copy_copy.at[idx, field] = dataDictionary_copy.at[idx, field]
            return dataDictionary_copy_copy

        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_copy[field].items():
                if dataDictionary_copy_mask.at[idx, field] == 1:
                    dataDictionary_copy_copy.at[idx, field] = np.NaN
            dataDictionary_copy_copy[field] = dataDictionary_copy_copy[field].interpolate(method='linear', limit_direction='both')
            # For each índex in the column
            for idx in dataDictionary_copy.index:
                # Verify if the value is NaN in the original dataframe
                if pd.isnull(dataDictionary_copy.at[idx, field]):
                    # Replace the value with the corresponding one from dataDictionary_copy_copy
                    dataDictionary_copy_copy.at[idx, field] = dataDictionary_copy.at[idx, field]
            return dataDictionary_copy_copy

    return dataDictionary_copy


def specialTypeMean(dataDictionary_copy: pd.DataFrame, specialTypeInput: SpecialType,
                    dataDictionary_copy_mask: pd.DataFrame = None,
                    missing_values: list = None, axis_param: int = None, field: str = None) -> pd.DataFrame:
    """
    Apply the mean to the missing values of a dataframe
    :param dataDictionary_copy: dataframe with the data
    :param specialTypeInput: special type to apply to the missing values
    :param dataDictionary_copy_mask: dataframe with the outliers
    :param missing_values: list of missing values
    :param axis_param: axis to apply the mean.
    :param field: field to apply the mean.

    :return: dataframe with the mean applied to the missing values
    """

    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calculate the mean of these numeric columns
                mean_value = only_numbers_df.mean().mean()
                # Replace the missing values with the mean of the entire DataFrame using lambda
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(lambda x: mean_value if (x in missing_values or pd.isnull(x)) else x))
            elif axis_param == 0:
                means = dataDictionary_copy.apply(
                    lambda col: col[col.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                        col.dtype, np.number) else None)
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                            lambda x: x if not (x in missing_values or pd.isnull(x)) else means[col])
            elif axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.T
                means = dataDictionary_copy.apply(
                    lambda row: row[row.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                        row.dtype, np.number) else None)
                for row in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                        dataDictionary_copy[row] = dataDictionary_copy[row].apply(
                            lambda x: x if not (x in missing_values or pd.isnull(x)) else means[row])
                dataDictionary_copy = dataDictionary_copy.T
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calculate the mean of these numeric columns
                mean_value = only_numbers_df.mean().mean()
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(lambda x: mean_value if (x in missing_values) else x))

            elif axis_param == 0:
                means = dataDictionary_copy.apply(lambda col: col[col.apply(lambda x:
                        np.issubdtype(type(x), np.number))].mean() if np.issubdtype(col.dtype, np.number) else None)
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                            lambda x: x if not (x in missing_values) else means[col])
            elif axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.T
                means = dataDictionary_copy.apply(lambda row: row[row.apply(lambda x:
                        np.issubdtype(type(x), np.number))].mean() if np.issubdtype(row.dtype, np.number) else None)
                for row in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                        dataDictionary_copy[row] = dataDictionary_copy[row].apply(
                            lambda x: x if not (x in missing_values) else means[row])
                dataDictionary_copy = dataDictionary_copy.T

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calculate the mean of these numeric columns
                mean_value = only_numbers_df.mean().mean()
                # Replace the missing values with the mean of the entire DataFrame using lambda
                for col_name in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col_name].items():
                        if np.issubdtype(type(value), np.number) and dataDictionary_copy_mask.at[idx, col_name] == 1:
                            dataDictionary_copy.at[idx, col_name] = mean_value
            if axis_param == 0: # Iterate over each column
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        mean=dataDictionary_copy[col].mean()
                        for idx, value in dataDictionary_copy[col].items():
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy.at[idx, col] = mean
            elif axis_param == 1: # Iterate over each row
                for idx, row in dataDictionary_copy.iterrows():
                    if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                        mean=dataDictionary_copy.loc[idx].mean()
                        for col in row.index:
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy.at[idx, col] = mean

    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        elif field in dataDictionary_copy.columns:
            if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                if specialTypeInput == SpecialType.MISSING:
                    mean = dataDictionary_copy[field].mean()
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: mean if x in missing_values else x)
                if specialTypeInput == SpecialType.INVALID:
                    mean = dataDictionary_copy[field].mean()
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: mean if x in missing_values else x)
                if specialTypeInput == SpecialType.OUTLIER:
                    mean=dataDictionary_copy[field].mean()
                    for idx, value in dataDictionary_copy[field].items():
                        if dataDictionary_copy_mask.at[idx, field] == 1:
                            dataDictionary_copy.at[idx, field] = mean
            else:
                raise ValueError("The field is not numeric")

    return dataDictionary_copy


def specialTypeMedian(dataDictionary_copy: pd.DataFrame, specialTypeInput: SpecialType,
                      dataDictionary_copy_mask: pd.DataFrame = None,
                      missing_values: list = None, axis_param: int = None, field: str = None) -> pd.DataFrame:
    """
    Apply the median to the missing values of a dataframe
    :param dataDictionary_copy: dataframe with the data
    :param specialTypeInput: special type to apply to the missing values
    :param missing_values: list of missing values
    :param axis_param: axis to apply the median.
    :param field: field to apply the median.

    :return: dataframe with the median applied to the missing values
    """
    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(lambda x: median_value if (x in missing_values or pd.isnull(x)) else x))
            elif axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        # Calculate the median without taking into account the nan values
                        median = dataDictionary_copy[col].median()
                        # Replace the values in missing_values, as well as the null values of python, by the median
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                            lambda x: median if x in missing_values or pd.isnull(x) else x)
            elif axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.T
                medians = dataDictionary_copy.apply(
                    lambda row: row[row.apply(lambda x: np.issubdtype(type(x), np.number))].median() if np.issubdtype(
                        row.dtype, np.number) else None)
                for row in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                        dataDictionary_copy[row] = dataDictionary_copy[row].apply(
                            lambda x: x if not (x in missing_values or pd.isnull(x)) else medians[row])
                dataDictionary_copy = dataDictionary_copy.T

        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Replace the values in missing_values (INVALID VALUES) by the median
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(lambda x: median_value if (x in missing_values) else x))


            elif axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        # Calculate the median without taking into account the nan values
                        median = dataDictionary_copy[col].median()
                        # Replace the values in missing_values, as well as the null values of python, by the median
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                            lambda x: median if x in missing_values else x)
            elif axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.T
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        # Calculate the median without taking into account the nan values
                        median = dataDictionary_copy[col].median()
                        # Replace the values in missing_values, as well as the null values of python, by the median
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                            lambda x: median if x in missing_values else x)
                dataDictionary_copy = dataDictionary_copy.T

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                # Select only columns with numeric data, including all numeric types (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calculate the median of these numeric columns
                median_value = only_numbers_df.median().median()
                # Replace the outliers with the median of the entire DataFrame using lambda
                for col_name in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col_name].items():
                        if dataDictionary_copy_mask.at[idx, col_name] == 1:
                            dataDictionary_copy.at[idx, col_name] = median_value

            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        median=dataDictionary_copy[col].median()
                        for idx, value in dataDictionary_copy[col].items():
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy.at[idx, col] = median
            elif axis_param == 1:
                for idx, row in dataDictionary_copy.iterrows():
                    median=dataDictionary_copy.loc[idx].median()
                    for col in row.index:
                        if dataDictionary_copy_mask.at[idx, col] == 1:
                            dataDictionary_copy.at[idx, col] = median
    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        elif field in dataDictionary_copy.columns:
            if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                if specialTypeInput == SpecialType.MISSING:
                    median = dataDictionary_copy[field].median()
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: median if x in missing_values or pd.isnull(x) else x)
                if specialTypeInput == SpecialType.INVALID:
                    median = dataDictionary_copy[field].median()
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: median if x in missing_values else x)
                if specialTypeInput == SpecialType.OUTLIER:
                    median = dataDictionary_copy[field].median()
                    for idx, value in dataDictionary_copy[field].items():
                        if dataDictionary_copy_mask.at[idx, field] == 1:
                            dataDictionary_copy.at[idx, field] = median
            else:
                raise ValueError("The field is not numeric")

    return dataDictionary_copy


def specialTypeClosest(dataDictionary_copy: pd.DataFrame, specialTypeInput: SpecialType,
                       dataDictionary_copy_mask: pd.DataFrame = None,
                       missing_values: list = None, axis_param: int = None, field: str = None) -> pd.DataFrame:
    """
    Apply the closest to the missing values of a dataframe
    :param dataDictionary_copy: dataframe with the data
    :param specialTypeInput: special type to apply to the missing values
    :param dataDictionary_copy_mask: dataframe with the outliers mask
    :param missing_values: list of missing values
    :param axis_param: axis to apply the closest value.
    :param field: field to apply the closest value.

    :return: dataframe with the closest applied to the missing values
    """

    if field is None:
        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Flatten the DataFrame into a single series of values
                flattened_values = only_numbers_df.values.flatten().tolist()

                # Create a dictionary to store the closest value for each missing value
                closest_values = {}

                # For each missing value, find the closest numeric value in the flattened series
                for missing_value in missing_values:
                    if missing_value not in closest_values:
                        closest_values[missing_value] = find_closest_value(flattened_values, missing_value)

                # Replace the missing values with the closest numeric values
                for i in range(len(dataDictionary_copy.index)):
                    for col, value in dataDictionary_copy.iloc[i].items():
                        current_value = dataDictionary_copy.at[i, col]
                        if current_value in closest_values:
                            dataDictionary_copy.at[i, col] = closest_values[current_value]
                        else:
                            if pd.isnull(dataDictionary_copy.at[i, col]) and specialTypeInput == SpecialType.MISSING:
                                raise ValueError(
                                    "Error: it's not possible to apply the closest operation to the null values")

            elif axis_param == 0:
                # Iterate over each column
                for col_name in dataDictionary_copy.select_dtypes(include=[np.number]).columns:
                    # Get the missing values in the current column
                    missing_values_in_col = [val for val in missing_values if val in dataDictionary_copy[col_name].values]

                    # If there are no missing values in the column, skip the rest of the loop
                    if not missing_values_in_col:
                        continue

                    # Flatten the column into a list of values
                    flattened_values = dataDictionary_copy[col_name].values.flatten().tolist()

                    # Create a dictionary to store the closest value for each missing value
                    closest_values = {}

                    # For each missing value IN the column (more efficient), find the closest numeric value in the flattened series
                    for missing_value in missing_values_in_col:
                        if missing_value not in closest_values:
                            closest_values[missing_value] = find_closest_value(flattened_values, missing_value)

                    # Replace the missing values with the closest numeric values in the column
                    for i in range(len(dataDictionary_copy.index)):
                        current_value = dataDictionary_copy.at[i, col_name]
                        if current_value in closest_values:
                            dataDictionary_copy.at[i, col_name] = closest_values[current_value]
                        else:
                            if pd.isnull(dataDictionary_copy.at[i, col_name]) and specialTypeInput == SpecialType.MISSING:
                                raise ValueError(
                                    "Error: it's not possible to apply the closest operation to the null values")
            elif axis_param == 1:
                # Iterate over each row
                for row_idx in range(len(dataDictionary_copy.index)):
                    # Get the numeric values in the current row
                    numeric_values_in_row = dataDictionary_copy.iloc[row_idx].select_dtypes(
                        include=[np.number]).values.tolist()

                    # Get the missing values in the current row
                    missing_values_in_row = [val for val in missing_values if val in numeric_values_in_row]

                    # If there are no missing values in the row, skip the rest of the loop
                    if not missing_values_in_row and not pd.isnull(dataDictionary_copy.iloc[row_idx]).any():
                        continue

                    # Create a dictionary to store the closest value for each missing value
                    closest_values = {}

                    # For each missing value IN the row (more efficient), find the closest numeric value in the numeric values
                    for missing_value in missing_values_in_row:
                        if missing_value not in closest_values:
                            closest_values[missing_value] = find_closest_value(numeric_values_in_row, missing_value)

                    # Replace the missing values with the closest numeric values in the row
                    for col_name in dataDictionary_copy.columns:
                        current_value = dataDictionary_copy.at[row_idx, col_name]
                        if current_value in closest_values:
                            dataDictionary_copy.at[row_idx, col_name] = closest_values[current_value]
                        else:
                            if pd.isnull(dataDictionary_copy.at[row_idx, col_name]) and specialTypeInput == SpecialType.MISSING:
                                raise ValueError("Error: it's not possible to apply the closest operation to the null values")

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Flatten the DataFrame into a single series of values
                flattened_values = only_numbers_df.values.flatten().tolist()

                # Create a dictionary to store the closest value for each outlier value
                closest_values = {}

                # For each outlier value, find the closest numeric value in the flattened series
                for i in range(len(dataDictionary_copy.index)):
                    for j in range(len(dataDictionary_copy.columns)):
                        if dataDictionary_copy_mask.iloc[i, j] == 1:
                            current_value = dataDictionary_copy.iloc[i, j]
                            if current_value not in closest_values:
                                closest_values[current_value] = find_closest_value(flattened_values, current_value)

                # Replace the outlier values with the closest numeric values
                for i in range(len(dataDictionary_copy.index)):
                    for j in range(len(dataDictionary_copy.columns)):
                        if dataDictionary_copy_mask.iloc[i, j] == 1:
                            current_value = dataDictionary_copy.iloc[i, j]
                            dataDictionary_copy.at[i, j] = closest_values[current_value]
            elif axis_param == 0:
                # Iterate over each column
                for col_name in dataDictionary_copy.select_dtypes(include=[np.number]).columns:
                    # Get the outlier values in the current column
                    outlier_values_in_col = [dataDictionary_copy.at[i, col_name] for i in
                                             range(len(dataDictionary_copy.index))
                                             if dataDictionary_copy_mask.at[i, col_name] == 1]

                    # If there are no outlier values in the column, skip the rest of the loop
                    if not outlier_values_in_col:
                        continue
                    # Flatten the column into a list of values
                    flattened_values = dataDictionary_copy[col_name].values.flatten().tolist()

                    # Create a dictionary to store the closest value for each outlier value
                    closest_values = {}

                    for outlier_value in outlier_values_in_col:
                        if outlier_value not in closest_values:
                            closest_values[outlier_value] = find_closest_value(flattened_values, outlier_value)

                    # Replace the outlier values with the closest numeric values in the column
                    for i in range(len(dataDictionary_copy.index)):
                        current_value = dataDictionary_copy.at[i, col_name]
                        if dataDictionary_copy_mask.at[i, col_name] == 1:
                            dataDictionary_copy.at[i, col_name] = closest_values[current_value]
            elif axis_param == 1:
                # Iterate over each row
                for row_idx in range(len(dataDictionary_copy.index)):
                    # Get the numeric values in the current row
                    numeric_values_in_row = dataDictionary_copy.iloc[row_idx].select_dtypes(
                        include=[np.number]).values.tolist()

                    # Get the outlier values in the current row
                    outlier_values_in_row = [dataDictionary_copy.at[row_idx, col_name] for col_name in
                                             dataDictionary_copy.columns
                                             if dataDictionary_copy_mask.at[row_idx, col_name] == 1]

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
                    for col_name in dataDictionary_copy.columns:
                        current_value = dataDictionary_copy.at[row_idx, col_name]
                        if dataDictionary_copy_mask.at[row_idx, col_name] == 1:
                            dataDictionary_copy.at[row_idx, col_name] = closest_values[current_value]

    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("Field not found in the dataDictionary_in")
        if not np.issubdtype(dataDictionary_copy[field].dtype, np.number):
            raise ValueError("Field is not numeric")

        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            # Get the missing values in the current column
            missing_values_in_col = [val for val in missing_values if val in dataDictionary_copy[field].values]

            # If there are no missing values in the column, skip the rest of the loop
            if missing_values_in_col or pd.isnull(dataDictionary_copy[field]).any():
                # Flatten the column into a list of values
                flattened_values = dataDictionary_copy[field].values.flatten().tolist()

                # Create a dictionary to store the closest value for each missing value
                closest_values = {}

                # For each missing value IN the column (more efficient), find the closest numeric value in the flattened series
                for missing_value in missing_values_in_col:
                    if missing_value not in closest_values:
                        closest_values[missing_value] = find_closest_value(flattened_values, missing_value)

                # Replace the missing values with the closest numeric values in the column
                for i in range(len(dataDictionary_copy.index)):
                    current_value = dataDictionary_copy.at[i, field]
                    if current_value in closest_values:
                        dataDictionary_copy.at[i, field] = closest_values[current_value]
                    else:
                        if pd.isnull(dataDictionary_copy.at[i, field]) and specialTypeInput == SpecialType.MISSING:
                            raise ValueError("Error: it's not possible to apply the closest operation to the null values")

        if specialTypeInput == SpecialType.OUTLIER:
            # Get the outlier values in the current column
            outlier_values_in_col = [dataDictionary_copy.at[i, field] for i in range(len(dataDictionary_copy.index))
                                     if dataDictionary_copy_mask.at[i, field] == 1]

            # If there are no outlier values in the column, skip the rest of the loop
            if outlier_values_in_col:
                # Flatten the column into a list of values
                flattened_values = dataDictionary_copy[field].values.flatten().tolist()

                # Create a dictionary to store the closest value for each outlier value
                closest_values = {}

                # For each outlier value IN the column (more efficient), find the closest numeric value in the flattened series
                for outlier_value in outlier_values_in_col:
                    if outlier_value not in closest_values:
                        closest_values[outlier_value] = find_closest_value(flattened_values, outlier_value)

                # Replace the outlier values with the closest numeric values in the column
                for i in range(len(dataDictionary_copy.index)):
                    current_value = dataDictionary_copy.at[i, field]
                    if dataDictionary_copy_mask.at[i, field] == 1:
                        dataDictionary_copy.at[i, field] = closest_values[current_value]

    return dataDictionary_copy
