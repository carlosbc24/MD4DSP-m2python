# Importing libraries
from datetime import datetime
# Importing functions and classes from packages
from typing import Union

import numpy as np
import pandas as pd

from helpers.auxiliar import compare_numbers, count_abs_frequency, getOutliers
from helpers.enumerations import Belong, Operator, Closure


class ContractsPrePost:
    def checkFieldRange(self, fields: list, dataDictionary: pd.DataFrame, belongOp: Belong) -> bool:
        """
        Check if fields meets the condition of belongOp in dataDictionary.
        If belongOp is Belong.BELONG, then it checks if all fields are in dataDictionary.
        If belongOp is Belong.NOTBELONG, then it checks if any field in 'fields' are not in dataDictionary.

        :param fields: list of columns
        :param dataDictionary: data dictionary
        :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG

        :return: if fields meets the condition of belongOp in dataDictionary
        :rtype: bool
        """
        if belongOp == Belong.BELONG:
            for field in fields:
                if field not in dataDictionary.columns:
                    return False  # Caso 1
            return True  # Caso 2
        elif belongOp == Belong.NOTBELONG:
            for field in fields:
                if field not in dataDictionary.columns:
                    return True  # Caso 3
            return False  # Caso 4

    def checkFixValueRange(self, value: Union[str, float, datetime], dataDictionary: pd.DataFrame, belongOp: Belong,
                           field: str = None,
                           quant_abs: int = None, quant_rel: float = None, quant_op: Operator = None) -> bool:
        """
        Check if fields meets the condition of belongOp in dataDictionary

        :param value: float value to check
        :param dataDictionary: data dictionary
        :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param field: dataset column in which value will be checked
        :param quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
               Operator.LESS=3, Operator.EQUAL=4
        :param quant_abs: integer which represents the absolute number of times that value should appear
                            with respect the enum operator quant_op
        :param quant_rel: float which represents the relative number of times that value should appear
                            with respect the enum operator quant_op

        :return: if fields meets the condition of belongOp in dataDictionary and field
        :rtype: bool
        """
        dataDictionary = dataDictionary.replace({
            np.nan: None})  # Se sustituyen los NaN por None para que no de error al hacer la comparacion de None con NaN. Como el dataframe es de floats, los None se convierten en NaN
        if value is not None and type(value) is not str and type(
                value) is not pd.Timestamp:  # Antes del casteo se debe comprobar que value no sea None, str o datetime(Timestamp), para que solo se casteen los int
            value = float(
                value)  # Se castea el valor a float para que no de error al hacer un get por valor, porque al hacer un get detecta el valor como int

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if value is in dataDictionary
                    return True if value in dataDictionary.values else False  # Caso 1 y 2
                else:
                    if quant_rel is not None and quant_abs is None:  # Check if value is in dataDictionary and if it meets the condition of quant_rel
                        return True if value in dataDictionary.values and compare_numbers(  # Caso 3 y 4
                            count_abs_frequency(value, dataDictionary) / dataDictionary.size,
                            quant_rel,
                            quant_op) else False  # Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                        # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                    elif quant_rel is not None and quant_abs is not None:
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError(
                            "quant_rel and quant_abs can't have different values than None at the same time")  # Caso 4.5
                    elif quant_abs is not None:
                        return True if value in dataDictionary.values and compare_numbers(  # Caso 5 y 6
                            count_abs_frequency(value, dataDictionary),
                            quant_abs,
                            quant_op) else False  # Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                    else:
                        raise ValueError(  # Caso 7
                            "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    return True if value not in dataDictionary.values else False  # Caso 8 y 9
                else:
                    raise ValueError(
                        "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Caso 10
        else:
            if field is not None:
                if field not in dataDictionary.columns:  # Se comprueba que la columna exista en el dataframe
                    raise ValueError(f"Column '{field}' not found in dataDictionary.")  # Caso 10.5
                if belongOp == Belong.BELONG:
                    if quant_op is None:
                        return True if value in dataDictionary[field].values else False  # Caso 11 y 12
                    else:
                        if quant_rel is not None and quant_abs is None:  # Añadido respecto a la especificacion inicial del contrato para test case 4
                            return True if value in dataDictionary[field].values and compare_numbers(  # Caso 13 y 14
                                dataDictionary[field].value_counts(dropna=False).get(value, 0)
                                / dataDictionary.size, quant_rel,
                                quant_op) else False  # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                        elif quant_rel is not None and quant_abs is not None:
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError(  # Caso 14.5
                                "quant_rel and quant_abs can't have different values than None at the same time")
                        elif quant_abs is not None:
                            return True if value in dataDictionary[field].values and compare_numbers(
                                dataDictionary[field].value_counts(dropna=False).get(value, 0),
                                quant_abs, quant_op) else False  # Caso 15 y 16
                        else:  # quant_rel is None and quant_abs is None
                            raise ValueError(  # Caso 17
                                "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        return True if value not in dataDictionary[field].values else False  # Caso 18 y 19
                    else:  # Caso 20
                        raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")

    def checkIntervalRangeFloat(self, left_margin: float, right_margin: float, dataDictionary: pd.DataFrame,
                                closureType: Closure, belongOp: Belong, field: str = None) -> bool:
        """
            Check if the dataDictionary meets the condition of belongOp in the interval defined by leftMargin and rightMargin with the closureType.
            If field is None, it does the check in the whole dataDictionary. If not, it does the check in the column specified by field.

            :param left_margin: float value which represents the left margin of the interval
            :param right_margin: float value which represents the right margin of the interval
            :param dataDictionary: data dictionary
            :param closureType: enum operator which can be Closure.openOpen=0, Closure.openClosed=1,
                                Closure.closedOpen=2, Closure.closedClosed=3
            :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
            :param field: dataset column in which value will be checked

            :return: if dataDictionary meets the condition of belongOp in the interval defined by leftMargin and rightMargin with the closureType
        """
        if left_margin > right_margin:
            raise ValueError("Error: leftMargin should be less than or equal to rightMargin")  # Caso 0

        def check_condition(min_val: float, max_val: float) -> bool:
            if closureType == Closure.openOpen:
                return True if (min_val > left_margin) & (max_val < right_margin) else False
            elif closureType == Closure.openClosed:
                return True if (min_val > left_margin) & (max_val <= right_margin) else False
            elif closureType == Closure.closedOpen:
                return True if (min_val >= left_margin) & (max_val < right_margin) else False
            elif closureType == Closure.closedClosed:
                return True if (min_val >= left_margin) & (max_val <= right_margin) else False
            else:
                raise ValueError("Error: closureType should be openOpen, openClosed, closedOpen, or closedClosed")

        if field is None:
            dataDictionary = dataDictionary.select_dtypes(
                include=['int', 'float'])  # Se descartan todos los campos que no sean float o int o double
            return check_condition(dataDictionary.min().min(),
                                   dataDictionary.max().max()) if belongOp == Belong.BELONG else not check_condition(
                dataDictionary.min().min(), dataDictionary.max().max())  # Casos 1-16
        else:
            if field not in dataDictionary.columns:  # Se comprueba que la columna exista en el dataframe
                raise ValueError(f"Column '{field}' not found in dataDictionary.")  # Caso 16.5
            if dataDictionary[field].dtype in ['int', 'float']:
                return check_condition(dataDictionary[field].min(),
                                       dataDictionary[
                                           field].max()) if belongOp == Belong.BELONG else not check_condition(
                    dataDictionary[field].min(), dataDictionary[field].max())  # Casos 17-32
            else:
                raise ValueError("Error: field should be a float")  # Caso 33

    def checkMissingRange(self, belongOp: Belong, dataDictionary: pd.DataFrame, field: str = None,
                          missing_values: list = None, quant_abs: int = None, quant_rel: float = None,
                          quant_op: Operator = None) -> bool:
        """
        Check if the dataDictionary meets the condition of belongOp with respect to the missing values defined in missing_values.
        If field is None, it does the check in the whole dataDictionary. If not, it does the check in the column specified by field.

        :param missing_values: list of missing values
        :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param dataDictionary: data dictionary
        :param field: dataset column in which value will be checked

        :return: if dataDictionary meets the condition of belongOp with respect to the missing values defined in missing_values
        """
        if missing_values is not None:
            for i in range(len(missing_values)):
                if isinstance(missing_values[i], int):
                    missing_values[i] = float(missing_values[i])

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if there are any missing values in dataDictionary
                    if dataDictionary.isnull().values.any():
                        return True  # Caso 1
                    else:  # If there aren't null python values in dataDictionary, it checks if there are any of the
                        # missing values in the list 'missing_values'
                        if missing_values is not None:
                            return True if any(
                                value in missing_values for value in
                                dataDictionary.values.flatten()) else False  # Caso 2 y 3
                        else:  # If the list is None, it returns False. It checks that in fact there aren't any missing values
                            return False  # Caso 4
                else:
                    if quant_rel is not None and quant_abs is None:  # Check there are any null python values or missing values from the list 'missing_values' in dataDictionary and if it meets the condition of quant_rel and quant_op
                        if (dataDictionary.isnull().values.any() or (missing_values is not None and any(
                                value in missing_values for value in
                                dataDictionary.values.flatten()))) and compare_numbers(
                            (dataDictionary.isnull().values.sum() + sum(
                                [count_abs_frequency(value, dataDictionary) for value in
                                 (missing_values if missing_values is not None else [])])) / dataDictionary.size,
                            quant_rel, quant_op):
                            return True  # Caso 5
                        else:
                            return False  # Caso 6
                    elif quant_rel is not None and quant_abs is not None:
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError(
                            "quant_rel and quant_abs can't have different values than None at the same time")  # Caso 7
                    elif quant_abs is not None:  # Check there are any null python values or missing values from the
                        # list 'missing_values' in dataDictionary and if it meets the condition of quant_abs and
                        # quant_op
                        if (dataDictionary.isnull().values.any() or (
                                missing_values is not None and any(
                            value in missing_values for value in dataDictionary.values.flatten()))) and compare_numbers(
                            dataDictionary.isnull().values.sum() + sum(
                                [count_abs_frequency(value, dataDictionary) for value in
                                 (missing_values if missing_values is not None else [])]),
                            quant_abs, quant_op):
                            return True  # Caso 8
                        else:
                            return False  # Caso 9
                    else:
                        raise ValueError(
                            "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is "
                            "not None")  # Caso 10
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    # Check that there aren't any null python values or missing values from the list 'missing_values'
                    # in dataDictionary
                    if missing_values is not None:
                        return True if not dataDictionary.isnull().values.any() and not any(
                            value in missing_values for value in
                            dataDictionary.values.flatten()) else False  # Caso 11 y 12
                    else:  # If the list is None, it checks that there aren't any python null values in dataDictionary
                        return True if not dataDictionary.isnull().values.any() else False  # Caso 13 y 13.5
                else:
                    raise ValueError(
                        "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Caso 14
        else:
            if field is not None:  # Comprobación de más añadida para que el código sea más legible
                if field not in dataDictionary.columns:  # Se comprueba que la columna exista en el dataframe
                    raise ValueError(f"Column '{field}' not found in dataDictionary.")  # Caso 15
                if belongOp == Belong.BELONG:
                    if quant_op is None:  # Check that there are null python values or missing values from the list
                        # 'missing_values' in the column specified by field
                        if dataDictionary[field].isnull().values.any():
                            return True  # Caso 16
                        else:  # If there aren't null python values in dataDictionary, it checks if there are any of the
                            # missing values in the list 'missing_values'
                            if missing_values is not None:
                                return True if any(
                                    value in missing_values for value in
                                    dataDictionary[field].values) else False  # Caso 17 y 18
                            else:  # If the list is None, it returns False. It checks that in fact there aren't any missing values
                                return False  # Caso 19
                    else:
                        if quant_rel is not None and quant_abs is None:  # Check there are null python values or
                            # missing values from the list 'missing_values' in the column specified by field and if it
                            # meets the condition of quant_rel and quant_op
                            if (dataDictionary[field].isnull().values.any() or (
                                    missing_values is not None and any(value in missing_values for value in
                                                                       dataDictionary[
                                                                           field].values))) and compare_numbers(
                                (dataDictionary[field].isnull().values.sum() + sum(
                                    [count_abs_frequency(value, dataDictionary, field) for value in
                                     (missing_values if missing_values is not None else [])])) / dataDictionary[
                                    field].size, quant_rel, quant_op):
                                return True  # Caso 20
                            else:
                                return False  # Caso 21
                            # Importante destacar que es la cantidad relativa respecto a los valores de la columna
                        elif quant_rel is not None and quant_abs is not None:
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError(
                                "quant_rel and quant_abs can't have different values than None at the same time")  # Caso 22
                        elif quant_abs is not None:  # Check there are null python values or missing values from the
                            # list 'missing_values' in the column specified by field and if it meets the condition of
                            # quant_abs and quant_op
                            if (dataDictionary[field].isnull().values.any() or (
                                    missing_values is not None and any(value in missing_values for value in
                                                                       dataDictionary[
                                                                           field].values))) and compare_numbers(
                                dataDictionary[field].isnull().values.sum() + sum(
                                    [count_abs_frequency(value, dataDictionary, field) for value in
                                     (missing_values if missing_values is not None else [])]),
                                quant_abs, quant_op):
                                return True  # Caso 23
                            else:
                                return False  # Caso 24
                        else:  # quant_rel is None and quant_abs is None
                            raise ValueError(
                                "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")  # Caso 25
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        # Check that there aren't any null python values or missing values from the list
                        # 'missing_values' in the column specified by field
                        if missing_values is not None:  # Check that there are missing values in the list 'missing_values'
                            return True if not dataDictionary[field].isnull().values.any() and not any(
                                value in missing_values for value in
                                dataDictionary[field].values) else False  # Caso 26 y 27
                        else:  # If the list is None, it checks that there aren't any python null values in the column specified by field
                            return True if not dataDictionary[field].isnull().values.any() else False  # Caso 28 y 29
                    else:
                        raise ValueError(
                            "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Caso 30

    def checkInvalidValues(self, belongOp: Belong, dataDictionary: pd.DataFrame, invalid_values: list,
                           field: str = None, quant_abs: int = None, quant_rel: float = None,
                           quant_op: Operator = None) -> bool:
        """
        Check if the dataDictionary meets the condition of belongOp with respect to the invalid values defined in invalid_values.
        If field is None, it does the check in the whole dataDictionary. If not, it does the check in the column specified by field.

        :param belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param dataDictionary: data dictionary
        :param invalid_values: list of invalid values
        :param field: dataset column in which value will be checked
        :param quant_abs: integer which represents the absolute number of times that value should appear
                            with respect the enum operator quant_op
        :param quant_rel: float which represents the relative number of times that value should appear
                            with respect the enum operator quant_op
        :param quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
               Operator.LESS=3, Operator.EQUAL=4

        :return: if dataDictionary meets the condition of belongOp with respect to the invalid values defined in invalid_values
        """
        if invalid_values is not None:
            for i in range(len(invalid_values)):
                if isinstance(invalid_values[i], int):
                    invalid_values[i] = float(invalid_values[i])

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if there are any invalid values in dataDictionary
                    if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                        if any(value in invalid_values for value in dataDictionary.values.flatten()):
                            return True  # Caso 1
                        else:
                            return False  # Caso 2
                    else:  # If the list is None, it returns False. It checks that in fact there aren't any invalid values
                        return False  # Caso 3
                else:
                    if quant_rel is not None and quant_abs is None:  # Check there are any invalid values in
                        # dataDictionary and if it meets the condition of quant_rel and quant_op (relative frequency)
                        if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                            if any(value in invalid_values for value in
                                   dataDictionary.values.flatten()) and compare_numbers(
                                sum([count_abs_frequency(value, dataDictionary) for value in
                                     invalid_values]) / dataDictionary.size, quant_rel, quant_op):
                                return True  # Caso 4
                            else:
                                return False  # Caso 5
                        else:  # If the list is None, it returns False
                            return False  # Caso 6
                    elif quant_abs is not None and quant_rel is None:  # Check there are any invalid values in
                        # dataDictionary and if it meets the condition of quant_abs and quant_op (absolute frequency)
                        if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                            if any(value in invalid_values for value in
                                   dataDictionary.values.flatten()) and compare_numbers(
                                sum([count_abs_frequency(value, dataDictionary) for value in
                                     invalid_values]), quant_abs, quant_op):
                                return True  # Caso 7
                            else:
                                return False  # Caso 8
                        else:  # If the list is None, it returns False
                            return False  # Caso 9
                    elif quant_abs is not None and quant_rel is not None:
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError(
                            "quant_rel and quant_abs can't have different values than None at the same time")  # Caso 10
                    else:
                        raise ValueError(
                            "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is "
                            "not None")  # Caso 11
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    # Check that there aren't any invalid values in dataDictionary
                    return True if not (invalid_values is not None and any(
                        value in invalid_values for value in
                        dataDictionary.values.flatten())) else False  # Caso 12 y 13
                else:
                    raise ValueError(
                        "Error: quant_op, quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Caso 14
        else:
            if field is not None:
                if field not in dataDictionary.columns:
                    raise ValueError(f"Column '{field}' not found in dataDictionary.")  # Caso 15
                if belongOp == Belong.BELONG:
                    if quant_op is None:  # Check that there are invalid values in the column specified by field
                        if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                            if any(value in invalid_values for value in dataDictionary[field].values):
                                return True  # Caso 16
                            else:
                                return False  # Caso 17
                        else:  # If the list is None, it returns False
                            return False  # Caso 18
                    else:
                        if quant_rel is not None and quant_abs is None:  # Check there are invalid values in the
                            # column specified by field and if it meets the condition of quant_rel and quant_op
                            # (relative frequency)
                            if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                                if any(value in invalid_values for value in
                                       dataDictionary[field].values) and compare_numbers(
                                    sum([count_abs_frequency(value, dataDictionary, field) for value in
                                         invalid_values]) / dataDictionary[field].size, quant_rel, quant_op):
                                    return True  # Caso 19
                                else:
                                    return False  # Caso 20
                            else:  # If the list is None, it returns False
                                return False  # Caso 21
                        elif quant_abs is not None and quant_rel is None:  # Check there are invalid values in the
                            # column specified by field and if it meets the condition of quant_abs and quant_op
                            # (absolute frequency)
                            if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                                if any(value in invalid_values for value in
                                       dataDictionary[field].values) and compare_numbers(
                                    sum([count_abs_frequency(value, dataDictionary, field) for value in
                                         invalid_values]), quant_abs, quant_op):
                                    return True  # Caso 22
                                else:
                                    return False  # Caso 23
                            else:  # If the list is None, it returns False
                                return False  # Caso 24
                        elif quant_abs is not None and quant_rel is not None:
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError(
                                "quant_rel and quant_abs can't have different values than None at the same time")  # Caso 25
                        else:
                            raise ValueError(
                                "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")  # Caso 26
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        # Check that there aren't any invalid values in the column specified by field
                        if invalid_values is not None:  # Check that there are invalid values in the list 'invalid_values'
                            return True if not any(
                                value in invalid_values for value in
                                dataDictionary[field].values) else False  # Caso 27 y 28
                        else:  # If the list is None, it returns True
                            return True  # Caso 29
                    else:
                        raise ValueError(
                            "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Caso 30

    def checkOutliers(self, dataDictionary: pd.DataFrame, belongOp: Belong = None, field: str = None,
                      quant_abs: int = None, quant_rel: float = None, quant_op: Operator = None) -> bool:
        """
        Check if there are outliers in the numeric columns of dataDictionary. The Outliers are calculated using the IQR method, so the outliers are the values that are
        below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR

        :param dataDictionary: dataframe with the data
        :param axis_param: axis to get the outliers. If axis_param is None, the outliers are calculated for the whole dataframe.
        If axis_param is 0, the outliers are calculated for each column. If axis_param is 1, the outliers are calculated for each row (although it is not recommended as it is not a common use case)

        :return: boolean indicating if there are outliers in the dataDictionary
        """
        dataDictionary_copy = dataDictionary.copy()
        outlier = 1  # 1 is the value that is going to be used to check if there are outliers in the dataframe

        if field is None:
            dataDictionary_copy = getOutliers(dataDictionary=dataDictionary_copy, field=None, axis_param=None)
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if there are any invalid values in dataDictionary
                    if outlier in dataDictionary_copy.values:
                        return True  # Caso 1
                    else:
                        return False  # Caso 2
                else:
                    if quant_rel is not None and quant_abs is None:  # Check there are any invalid values in
                        # dataDictionary and if it meets the condition of quant_rel and quant_op (relative frequency)
                        if compare_numbers(count_abs_frequency(outlier, dataDictionary_copy) / dataDictionary_copy.size,
                                           quant_rel, quant_op):
                            return True  # Caso 3
                        else:
                            return False  # Caso 4
                    elif quant_abs is not None and quant_rel is None:  # Check there are any invalid values in
                        # dataDictionary and if it meets the condition of quant_abs and quant_op (absolute frequency)
                        if compare_numbers(count_abs_frequency(outlier, dataDictionary_copy), quant_abs, quant_op):
                            return True  # Caso 5
                        else:
                            return False  # Caso 6
                    elif quant_abs is not None and quant_rel is not None:
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError(
                            "quant_rel and quant_abs can't have different values than None at the same time")  # Caso 7
                    else:
                        raise ValueError(
                            "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is "
                            "not None")  # Caso 8
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    return True if not (outlier in dataDictionary_copy.values) else False  # Caso 9 y 10
                else:
                    raise ValueError("Error: quant_op, quant_rel and quant_abs should be None when belongOp is "
                                     "NOTBELONG")  # Caso 11
        else:
            if field is not None:
                if field not in dataDictionary.columns:
                    raise ValueError(f"Column '{field}' not found in dataDictionary.")  # Caso 12

                dataDictionary_copy = getOutliers(dataDictionary=dataDictionary_copy, field=field, axis_param=None)
                if belongOp == Belong.BELONG:
                    if quant_op is None:  # Check that there are invalid values in the column specified by field
                        if outlier in dataDictionary_copy[field].values:
                            return True  # Caso 13
                        else:
                            return False  # Caso 14
                    else:
                        if quant_rel is not None and quant_abs is None:  # Check there are invalid values in the
                            # column specified by field and if it meets the condition of quant_rel and quant_op
                            # (relative frequency)
                            if compare_numbers(
                                    count_abs_frequency(outlier, dataDictionary_copy) / dataDictionary_copy[field].size,
                                    quant_rel, quant_op):
                                return True  # Caso 15
                            else:
                                return False  # Caso 16
                        elif quant_abs is not None and quant_rel is None:  # Check there are invalid values in the
                            # column specified by field and if it meets the condition of quant_abs and quant_op
                            # (absolute frequency)
                            if compare_numbers(count_abs_frequency(outlier, dataDictionary_copy), quant_abs, quant_op):
                                return True  # Caso 17
                            else:
                                return False  # Caso 18
                        elif quant_abs is not None and quant_rel is not None:
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError(
                                "quant_rel and quant_abs can't have different values than None at the same time")  # Caso 19
                        else:
                            raise ValueError(
                                "Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")  # Caso 20
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        # Check that there aren't any invalid values in the column specified by field
                        return True if not (outlier in dataDictionary_copy[field].values) else False  # Caso 21 y 22
                    else:
                        raise ValueError(
                            "Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")  # Caso 23
