import numpy as np
# Importing libraries
import pandas as pd

# Importing enumerations from packages
from typing import Union
from helpers.enumerations import Operator, DataType, Closure, DerivedType, SpecialType


def compare_numbers(rel_abs_number: Union [int,float], quant_rel_abs: Union[int, float], quant_op: Operator) -> bool:
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


def count_abs_frequency(value, dataDictionary: pd.DataFrame, field: str=None) -> int:
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

def cast_type_FixValue(dataTypeInput: DataType=None, FixValueInput=None, dataTypeOutput: DataType=None, FixValueOutput=None):
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



def find_closest_value(numeric_values : list, value: Union[int, float]) -> Union[int, float]:
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



def getOutliers(dataDictionary_copy: pd.DataFrame, axis_param: int = None) -> pd.DataFrame:
    """
    Get the outliers of a dataframe
    :param dataDictionary_copy: dataframe with the data
    :param axis_param: axis to get the outliers. If axis_param is None, the outliers are calculated for the whole dataframe.
    If axis_param is 0, the outliers are calculated for each column. If axis_param is 1, the outliers are calculated for each row.

    :return: dataframe with the outliers. The value 1 indicates that the value is an outlier and the value 0 indicates that the value is not an outlier

    """
    dataDictionary_copy_copy = dataDictionary_copy.copy()
    threshold = 1.5
    if axis_param is None:
        Q1 = dataDictionary_copy_copy.stack().quantile(0.25)
        Q3 = dataDictionary_copy_copy.stack().quantile(0.75)
        IQR = Q3 - Q1
        # Definir los límites para identificar outliers
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        # Pone a 1 los valores que son outliers y a 0 los que no lo son
        for col in dataDictionary_copy_copy.columns:
            for idx, value in dataDictionary_copy_copy[col].items():
                if value < lower_bound or value > upper_bound:
                    dataDictionary_copy_copy.at[idx, col] = 1
                else:
                    dataDictionary_copy_copy.at[idx, col] = 0
        return dataDictionary_copy_copy

    elif axis_param == 0:
        for col in dataDictionary_copy_copy.columns:
            Q1 = dataDictionary_copy_copy[col].quantile(0.25)
            Q3 = dataDictionary_copy_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            # Definir los límites para identificar outliers
            lower_bound_col = Q1 - threshold * IQR
            upper_bound_col = Q3 + threshold * IQR

            for idx, value in dataDictionary_copy_copy[col].items():
                if value < lower_bound_col or value > upper_bound_col:
                    dataDictionary_copy_copy.at[idx, col] = 1
                else:
                    dataDictionary_copy_copy.at[idx, col] = 0
        return dataDictionary_copy_copy

    elif axis_param == 1:
        for idx, row in dataDictionary_copy_copy.iterrows():
            Q1 = row.quantile(0.25)
            Q3 = row.quantile(0.75)
            IQR = Q3 - Q1
            # Definir los límites para identificar outliers
            lower_bound_row = Q1 - threshold * IQR
            upper_bound_row = Q3 + threshold * IQR

            for col in row.index:
                value = row[col]
                if value < lower_bound_row or value > upper_bound_row:
                    dataDictionary_copy_copy.at[idx, col] = 1
                else:
                    dataDictionary_copy_copy.at[idx, col] = 0
        return dataDictionary_copy_copy


def apply_derivedTypeOutliers(derivedTypeOutput: DerivedType, dataDictionary_copy: pd.DataFrame, dataDictionary_copy_copy: pd.DataFrame, axis_param: int = None):
    if derivedTypeOutput == DerivedType.MOSTFREQUENT:
        if axis_param == 0:
            for col in dataDictionary_copy.columns:
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
                        dataDictionary_copy.at[idx, col] = dataDictionary_copy.at[idx-1, col]
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

    return dataDictionary_copy


def apply_derivedType(specialTypeInput: SpecialType,derivedTypeOutput: DerivedType, dataDictionary_copy: pd.DataFrame, missing_values: list = None,
                      axis_param: int = None) -> pd.DataFrame:
    if derivedTypeOutput == DerivedType.MOSTFREQUENT:
        if axis_param == 0:
            if specialTypeInput == SpecialType.MISSING:
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(
                        lambda x: dataDictionary_copy[col.name].value_counts().idxmax() if pd.isnull(x) else x))
            if missing_values is not None:
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.apply(
                        lambda x: dataDictionary_copy[col.name].value_counts().idxmax() if x in missing_values else x))
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
                                        if value in missing_values or pd.isnull(value) and i > 0 else value for i, value
                                                in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
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
                                            if (value in missing_values or pd.isnull(value)) and i < len(
                                                row_or_col) - 1 else value for i, value in enumerate(row_or_col)],
                                                          index=row_or_col.index), axis=axis_param)
            else:
                dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                    value) else row_or_col.iloc[i + 1] if value in missing_values and i < len(row_or_col) - 1 else value
                                                                                              for i, value in
                                                                                              enumerate(row_or_col)],
                                                                                             index=row_or_col.index),
                                                                axis=axis_param)
        elif axis_param is None:
            raise ValueError("The axis cannot be None when applying the NEXT operation")

    return dataDictionary_copy













