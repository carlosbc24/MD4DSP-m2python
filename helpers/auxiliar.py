# Importing enumerations from packages
from typing import Union

import numpy as np
# Importing libraries
import pandas as pd

from helpers.enumerations import Operator, DataType, DerivedType, SpecialType


def compare_numbers(rel_abs_number: Union[int, float], quant_rel_abs: Union[int, float], quant_op: Operator) -> bool:
    """
    Compare two numbers with the operator quant_op

    :param rel_abs_number: relative or absolute number to compare
    :param quant_rel_abs: relative or absolute number to compare with the previous one
    :param quant_op: quantifier operator to compare the two numbers

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
    :param dataDictionary: dataframe with the data
    :param field: column to count the value

    :return: absolute frequency of the value in the column
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
    :param value: value to compare with the list of numeric values

    :return: the closest value to the given value in the list of numeric values

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
    # Filtrar el DataFrame para incluir solo columnas numéricas
    dataDictionary_numeric = dataDictionary.select_dtypes(include=[np.number])

    dataDictionary_copy = dataDictionary.copy()
    # Inicializar todos los valores del DataFrame copiado a 0
    dataDictionary_copy.loc[:, :] = 0

    threshold = 1.5
    if field is None:
        if axis_param is None:
            Q1 = dataDictionary_numeric.stack().quantile(0.25)
            Q3 = dataDictionary_numeric.stack().quantile(0.75)
            IQR = Q3 - Q1
            # Definir los límites para identificar outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            # Pone a 1 los valores que son outliers y a 0 los que no lo son
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
                # Definir los límites para identificar outliers
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
                # Definir los límites para identificar outliers
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
                            prev_col = row.index[row.index.get_loc(col) - 1]  # Obtener la columna anterior
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
                      missing_values: list = None,
                      axis_param: int = None, field: str = None) -> pd.DataFrame:
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
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: dataDictionary_copy[col.name].value_counts().idxmax() if pd.isnull(x) else x))
                if missing_values is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: dataDictionary_copy[
                                col.name].value_counts().idxmax() if x in missing_values else x))
            elif axis_param == 1:
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda row: row.apply(
                            lambda x: dataDictionary_copy.loc[row.name].value_counts().idxmax() if pd.isnull(x) else x),
                        axis=axis_param)
                if missing_values is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda row: row.apply(lambda x: dataDictionary_copy.loc[
                            row.name].value_counts().idxmax() if x in missing_values else x), axis=axis_param)
            elif axis_param is None:
                valor_mas_frecuente = dataDictionary_copy.stack().value_counts().idxmax()
                if missing_values is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(lambda x: valor_mas_frecuente if x in missing_values else x))
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: dataDictionary_copy[col.name].value_counts().idxmax() if pd.isnull(x) else x))

        elif derivedTypeOutput == DerivedType.PREVIOUS:
            # Aplica la función lambda a nivel de columna (axis=0) o a nivel de fila (axis=1)
            if axis_param == 0 or axis_param == 1:
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([row_or_col.iloc[i - 1]
                                          if (value in missing_values or pd.isnull(value)) and i > 0 else value
                                           for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else:
                    # Define la función lambda para reemplazar los valores dentro de missing values por el valor de la posición anterior
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i - 1] if value in missing_values and i > 0 else value
                              for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)

            elif axis_param is None:
                raise ValueError("The axis cannot be None when applying the PREVIOUS operation")

        elif derivedTypeOutput == DerivedType.NEXT:
            # Define la función lambda para reemplazar los valores dentro de missing values por el valor de la siguiente posición
            if axis_param == 0 or axis_param == 1:
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([row_or_col.iloc[i + 1]
                                             if (value in missing_values or pd.isnull(value)) and i < len(row_or_col) - 1
                                                else value for i, value in enumerate(row_or_col)], index=row_or_col.index),
                                                    axis=axis_param)
                else:
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
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: dataDictionary_copy[field].value_counts().idxmax() if pd.isnull(x) else x)
                if missing_values is not None:
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: dataDictionary_copy[field].value_counts().idxmax() if x in missing_values else x)
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
                    if missing_values is not None:
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
            # Aplicamos la interpolación lineal en el DataFrame
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                            lambda x: np.nan if x in missing_values else x).interpolate(method='linear',
                                                                                        limit_direction='both')
            elif axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda row: row.apply(lambda x: np.nan if x in missing_values else x).interpolate(
                        method='linear', limit_direction='both'), axis=axis_param)

        if specialTypeInput == SpecialType.INVALID:
            # Aplicamos la interpolación lineal en el DataFrame
            if axis_param == 0:
                for col in dataDictionary_copy_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        dataDictionary_copy_copy[col] = dataDictionary_copy_copy[col].apply(
                            lambda x: np.nan if x in missing_values else x).interpolate(method='linear',
                                                                                        limit_direction='both')
            elif axis_param == 1:
                dataDictionary_copy_copy = dataDictionary_copy_copy.apply(
                    lambda row: row.apply(lambda x: np.nan if x in missing_values else x).interpolate(
                        method='linear', limit_direction='both'), axis=axis_param)

            # Verificamos si hay algún valor nulo en el DataFrame
            if dataDictionary_copy.isnull().any().any():
                dataDictionary_copy = dataDictionary_copy.apply(lambda row: row.apply(
                    lambda value: dataDictionary_copy_copy.at[row.name, value] if not pd.isnull(value) else value),
                                                                axis=1)
            else:
                dataDictionary_copy = dataDictionary_copy_copy.copy()

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        for idx, value in dataDictionary_copy[col].items():
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy.at[idx, col] = np.NaN
                                dataDictionary_copy[col] = dataDictionary_copy[col].interpolate(method='linear',
                                                                                                limit_direction='both')
            elif axis_param == 1:
                for idx, row in dataDictionary_copy.iterrows():
                    if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                        for col in row.index:
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy.at[idx, col] = np.NaN
                                dataDictionary_copy.at[idx, col] = dataDictionary_copy.loc[idx].interpolate(method='linear',
                                                                                                            limit_direction='both')
    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        if not np.issubdtype(dataDictionary_copy[field].dtype, np.number):
            raise ValueError("The field is not numeric")

        if specialTypeInput == SpecialType.MISSING:
            dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                lambda x: np.nan if x in missing_values else x).interpolate(method='linear', limit_direction='both')

        if specialTypeInput == SpecialType.INVALID:
            dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                lambda x: np.nan if x in missing_values else x).interpolate(method='linear', limit_direction='both')

            if dataDictionary_copy.isnull().any().any():
                dataDictionary_copy = dataDictionary_copy.apply(lambda row: row.apply(
                    lambda value: dataDictionary_copy_copy.at[row.name, value] if not pd.isnull(value) else value),
                                                                axis=1)
            else:
                dataDictionary_copy = dataDictionary_copy_copy.copy()

        if specialTypeInput == SpecialType.OUTLIER:
            for idx, value in dataDictionary_copy[field].items():
                if dataDictionary_copy_mask.at[idx, field] == 1:
                    dataDictionary_copy.at[idx, field] = np.NaN
                    dataDictionary_copy[field] = dataDictionary_copy[field].interpolate(method='linear',
                                                                                        limit_direction='both')

    return dataDictionary_copy


def specialTypeMean(dataDictionary_copy: pd.DataFrame, specialTypeInput: SpecialType,
                    dataDictionary_copy_mask: pd.DataFrame = None,
                    missing_values: list = None, axis_param: int = None, field: str = None) -> pd.DataFrame:
    """
    Apply the mean to the missing values of a dataframe
    :param dataDictionary_copy: dataframe with the data
    :param specialTypeInput: special type to apply to the missing values
    :param missing_values: list of missing values
    :param axis_param: axis to apply the mean.
    :param field: field to apply the mean.

    :return: dataframe with the mean applied to the missing values
    """

    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param == None:
                # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calcular la media de estas columnas numéricas
                mean_value = only_numbers_df.mean().mean()
                # Reemplaza 'fixValueInput' con la media del DataFrame completo usando lambda
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
            if axis_param == None:
                # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calcular la media de estas columnas numéricas
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
                # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calcular la media de estas columnas numéricas
                mean_value = only_numbers_df.mean().mean()
                # Reemplaza los outliers con la media del DataFrame completo usando lambda
                for col_name in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col_name].items():
                        if np.issubdtype(type(value), np.number) and dataDictionary_copy_mask.at[idx, col_name] == 1:
                            dataDictionary_copy.at[idx, col_name] = mean_value
            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        for idx, value in dataDictionary_copy[col].items():
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy[col] = dataDictionary_copy[col].mean()
            elif axis_param == 1:
                for idx, row in dataDictionary_copy.iterrows():
                    if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                        for col in row.index:
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy.at[idx, col] = dataDictionary_copy.loc[idx].mean()

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
                    for idx, value in dataDictionary_copy[field].items():
                        if dataDictionary_copy_mask.at[idx, field] == 1:
                            dataDictionary_copy.at[idx, field] = dataDictionary_copy[field].mean()
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
            if axis_param == None:
                # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calcular la media de estas columnas numéricas
                median_value = only_numbers_df.median().median()
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(
                        lambda x: median_value if (x in missing_values or pd.isnull(x)) else x))
            elif axis_param == 0 or axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(
                        lambda x: x if not (x in missing_values or pd.isnull(x))
                            else col[col.apply(lambda z: np.issubdtype(type(z), np.number))].median()), axis=axis_param)
        if specialTypeInput == SpecialType.INVALID:
            if axis_param == None:
                # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calcular la media de estas columnas numéricas
                median_value = only_numbers_df.median().median()
                # Reemplaza 'fixValueInput' con la media del DataFrame completo usando lambda
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(lambda x: median_value if (x in missing_values) else x))

            elif axis_param == 0 or axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(
                        lambda x: x if not (x in missing_values)
                            else col[col.apply(lambda z: np.issubdtype(type(z), np.number))].median()), axis=axis_param)

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                # Calcular la media de estas columnas numéricas
                median_value = only_numbers_df.median().median()
                # Reemplaza los outliers con la media del DataFrame completo usando lambda
                for col_name in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col_name].items():
                        if dataDictionary_copy_mask.at[idx, col_name] == 1:
                            dataDictionary_copy.at[idx, col_name] = median_value

            if axis_param == 0:
                for col in dataDictionary_copy.columns:
                    if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                        for idx, value in dataDictionary_copy[col].items():
                            if dataDictionary_copy_mask.at[idx, col] == 1:
                                dataDictionary_copy[col] = dataDictionary_copy[col].median()
            elif axis_param == 1:
                for idx, row in dataDictionary_copy.iterrows():
                    for col in row.index:
                        if dataDictionary_copy_mask.at[idx, col] == 1:
                            dataDictionary_copy.at[idx, col] = dataDictionary_copy.loc[idx].median()

    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        elif field in dataDictionary_copy.columns:
            if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: dataDictionary_copy[field].median() if x in missing_values else x)
                if specialTypeInput == SpecialType.INVALID:
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                        lambda x: dataDictionary_copy[field].median() if x in missing_values else x)
                if specialTypeInput == SpecialType.OUTLIER:
                    for idx, value in dataDictionary_copy[field].items():
                        if dataDictionary_copy_mask.at[idx, field] == 1:
                            dataDictionary_copy.at[idx, field] = dataDictionary_copy[field].median()
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
    :param missing_values: list of missing values
    :param axis_param: axis to apply the closest value.
    :param field: field to apply the closest value.

    :return: dataframe with the closest applied to the missing values
    """

    def raise_error():
        raise ValueError("Error: it's not possible to apply the closest operation to the null values")

    if field is None:
        if specialTypeInput == SpecialType.MISSING:
            if axis_param is None:
                dataDictionary_copy = dataDictionary_copy.apply(lambda col: col.apply(lambda x:
                                       find_closest_value(dataDictionary_copy.stack(), x) if x in missing_values else
                                                                raise_error() if pd.isnull(x) else x))
            elif axis_param == 0 or axis_param == 1:
                # Reemplazar los valores en missing_values por el valor numérico más cercano a lo largo de las columnas y filas
                dataDictionary_copy = dataDictionary_copy.apply(lambda col: col.apply(
                    lambda x: find_closest_value(col, x) if x in missing_values else raise_error() if pd.isnull(x)
                    else x), axis=axis_param)
        if specialTypeInput == SpecialType.INVALID:
            if axis_param is None:
                dataDictionary_copy = dataDictionary_copy.apply(lambda col: col.apply(lambda x: find_closest_value(
                                                    dataDictionary_copy.stack(), x) if x in missing_values else x))
            elif axis_param == 0 or axis_param == 1:
                # Reemplazar los valores en missing_values por el valor numérico más cercano a lo largo de las columnas y filas
                dataDictionary_copy = dataDictionary_copy.apply(lambda col: col.apply(
                    lambda x: find_closest_value(col, x) if x in missing_values else x), axis=axis_param)

        if specialTypeInput == SpecialType.OUTLIER:
            if axis_param is None:
                for col_name in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col_name].items():
                        if dataDictionary_copy_mask.at[idx, col_name] == 1:
                            dataDictionary_copy.at[idx, col_name] = find_closest_value(dataDictionary_copy.stack(), value)
            elif axis_param == 0 or axis_param == 1:
                # Reemplazar los valores en la misma posicion que los 1 dataDictionary_copy_mask por el valor numérico más cercano a lo largo de las columnas y filas
                for col_name in dataDictionary_copy.columns:
                    for idx, value in dataDictionary_copy[col_name].items():
                        if dataDictionary_copy_mask.at[idx, col_name] == 1:
                            dataDictionary_copy.at[idx, col_name] = find_closest_value(dataDictionary_copy[col_name], value)
    elif field is not None:
        if field not in dataDictionary_copy.columns:
            raise ValueError("The field is not in the dataframe")
        elif field in dataDictionary_copy.columns:
            if specialTypeInput == SpecialType.MISSING:
                dataDictionary_copy[field] = dataDictionary_copy[field].apply(lambda x: find_closest_value(
                                            dataDictionary_copy[field], x) if x in missing_values else raise_error()
                                                if pd.isnull(x) else x)
            if specialTypeInput == SpecialType.INVALID:
                dataDictionary_copy[field] = dataDictionary_copy[field].apply(lambda x: find_closest_value(
                                                    dataDictionary_copy[field], x) if x in missing_values else x)
            if specialTypeInput == SpecialType.OUTLIER:
                for idx, value in dataDictionary_copy[field].items():
                    if dataDictionary_copy_mask.at[idx, field] == 1:
                        dataDictionary_copy.at[idx, field] = find_closest_value(dataDictionary_copy[field], value)

    return dataDictionary_copy
