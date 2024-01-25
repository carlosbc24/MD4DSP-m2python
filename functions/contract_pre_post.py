from datetime import datetime

import numpy as np
import pandas as pd

from helpers.auxiliar import compare_numbers
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
                    return False
            return True
        elif belongOp == Belong.NOTBELONG:
            for field in fields:
                if field not in dataDictionary.columns:
                    return True
            return False

    def checkFixValueRangeString(self, value: str, dataDictionary: pd.DataFrame, belongOp: Belong, field: str = None,
                                 quant_abs: int = None, quant_rel: float = None, quant_op: Operator = None) -> bool:
        """
        Check if fields meets the condition of belongOp in dataDictionary

        :param value: string value to check
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
        dataDictionary = dataDictionary.replace({np.nan: None})
        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if value is in dataDictionary
                    return True if value in dataDictionary.values else False
                else:
                    if quant_rel is not None and quant_abs is None:  # Check if value is in dataDictionary and if it meets the condition of quant_rel
                        return True if value in dataDictionary.values and compare_numbers(
                                dataDictionary.value_counts(dropna=False).get(value, 0)
                                / dataDictionary.size, quant_rel, quant_op) else False  # Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                                # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                    elif quant_rel is not None and quant_abs is not None:
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError("Error: quant_rel and quant_abs can't have different values than None at the same time")
                    elif quant_abs is not None:
                        return True if value in dataDictionary.values and compare_numbers(
                                dataDictionary.value_counts(dropna=False).get(value, 0),
                                quant_abs, quant_op) else False  # Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                    else:
                        raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    return True if value not in dataDictionary.values else False
                else:
                    raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")
        else:
            if field is not None:
                if belongOp == Belong.BELONG:
                    if quant_op is None:
                        return True if value in dataDictionary[field].values else False
                    else:
                        if quant_rel is not None and quant_abs is None:  # Añadido respecto a la especificacion inicial del contrato para test case 4
                            return True if value in dataDictionary[field].values and compare_numbers(
                                    dataDictionary[field].value_counts(dropna=False).get(value, 0)
                                    / dataDictionary.size, quant_rel, quant_op) else False
                                    # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                        elif quant_rel is not None and quant_abs is not None:
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError("Error: quant_rel and quant_abs can't have different values than None at the same time")
                        elif quant_abs is not None:
                            return True if value in dataDictionary[field].values and compare_numbers(
                                    dataDictionary[field].value_counts(dropna=False).get(value, 0),
                                    quant_abs, quant_op) else False
                        else:  # quant_rel is None and quant_abs is None
                            raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        return True if value not in dataDictionary[field].values else False
                    else:
                        raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")

    def checkFixValueRangeFloat(self, value: float, dataDictionary: pd.DataFrame, belongOp: Belong, field: str = None,
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
        dataDictionary = dataDictionary.replace({np.nan: None})  # Se sustituyen los NaN por None para que no de error al hacer la comparacion de None con NaN. Como el dataframe es de floats, los None se convierten en NaN
        if value is not None:  # Se castea el valor a float para que no de error al hacer un get por valor, porque al hacer un get detecta el valor como int
            value = float(value)

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if value is in dataDictionary
                    return True if value in dataDictionary.values else False
                else:
                    if quant_rel is not None and quant_abs is None:  # Check if value is in dataDictionary and if it meets the condition of quant_rel
                        return True if value in dataDictionary.values and compare_numbers(
                                dataDictionary.value_counts(dropna=False).get(value, 0)
                                / dataDictionary.size, quant_rel, quant_op) else False  # Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                                # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                    elif quant_rel is not None and quant_abs is not None:
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError("quant_rel and quant_abs can't have different values than None at the same time")
                    elif quant_abs is not None:
                        return True if value in dataDictionary.values and compare_numbers(
                                dataDictionary.value_counts(dropna=False).get(value, 0),
                                quant_abs, quant_op) else False  # Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                    else:
                        raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    return True if value not in dataDictionary.values else False
                else:
                    raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")
        else:
            if field is not None:
                if belongOp == Belong.BELONG:
                    if quant_op is None:
                        return True if value in dataDictionary[field].values else False
                    else:
                        if quant_rel is not None and quant_abs is None:  # Añadido respecto a la especificacion inicial del contrato para test case 4
                            return True if value in dataDictionary[field].values and compare_numbers(
                                    dataDictionary[field].value_counts(dropna=False).get(value, 0)
                                    / dataDictionary.size, quant_rel, quant_op) else False  # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                        elif quant_rel is not None and quant_abs is not None:
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError("quant_rel and quant_abs can't have different values than None at the same time")
                        elif quant_abs is not None:
                            return True if value in dataDictionary[field].values and compare_numbers(
                                    dataDictionary[field].value_counts(dropna=False).get(value, 0),
                                    quant_abs, quant_op) else False
                        else:  # quant_rel is None and quant_abs is None
                            raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        return True if value not in dataDictionary[field].values else False
                    else:
                        raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")

    def checkFixValueRangeDateTime(self, value: datetime, dataDictionary: pd.DataFrame, belongOp: Belong,
                                   field: str = None, quant_abs: int = None, quant_rel: float = None,
                                   quant_op: Operator = None) -> bool:
        """
        Check if fields meets the condition of belongOp in dataDictionary

        :param value: datetime value to check
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

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if value is in dataDictionary
                    return True if value in dataDictionary.values else False
                else:
                    if quant_rel is not None and quant_abs is None:  # Check if value is in dataDictionary and if it meets the condition of quant_rel
                        return True if value in dataDictionary.values and compare_numbers(
                                dataDictionary.value_counts(dropna=False).get(value, 0)
                                / dataDictionary.size, quant_rel,
                                quant_op) else False  # Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                                                      # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                    elif quant_rel is not None and quant_abs is not None:
                        # Añadido respecto a la especificacion inicial del contrato para test case 4
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError("Error: quant_rel and quant_abs should be None at the same time")
                    elif quant_abs is not None:
                        return True if value in dataDictionary.values and compare_numbers(
                                dataDictionary.value_counts(dropna=False).get(value, 0),
                                quant_abs,quant_op) else False  # Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                    else:
                        raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    return True if value not in dataDictionary.values else False
                else:
                    raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")
        else:
            if field is not None:
                if belongOp == Belong.BELONG:
                    if quant_op is None:
                        return True if value in dataDictionary[field].values else False
                    else:
                        if quant_rel is not None and quant_abs is None:  # Añadido respecto a la especificacion inicial del contrato para test case 4
                            return True if value in dataDictionary[field].values and compare_numbers(
                                    dataDictionary[field].value_counts(dropna=False).get(value, 0)
                                    / dataDictionary.size, quant_rel, quant_op) else False      # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                        elif quant_rel is not None and quant_abs is not None:
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError("Error: quant_rel and quant_abs should be None at the same time")
                        elif quant_abs is not None:
                            return True if value in dataDictionary[field].values and compare_numbers(
                                    dataDictionary[field].value_counts(dropna=False).get(value, 0),
                                    quant_abs, quant_op) else False
                        else:  # quant_rel is None and quant_abs is None
                            raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        return True if value not in dataDictionary[field].values else False
                    else:
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
        """
        Si field es None: 
            Comprueba si el rango establecido a través de los valores rightMargin y leftMargin y el operador closureType cumplen con la condición definida por el operador BelongOp en el dataset de entrada. 
        Si no: 
            Comprueba si el rango establecido a través de los valores rightMargin y leftMargin y el operador closureType cumplen con la condición definida por el operador BelongOp en la columna especificada. 
        """

        if left_margin > right_margin:
            raise ValueError("Error: leftMargin should be less than or equal to rightMargin")

        def check_condition(min_val, max_val):
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
            return check_condition(dataDictionary.min().min(),
                                   dataDictionary.max().max()) if belongOp == Belong.BELONG else not check_condition(
                                   dataDictionary.min().min(), dataDictionary.max().max())
        else:
            return check_condition(dataDictionary[field].min(),
                                   dataDictionary[field].max()) if belongOp == Belong.BELONG else not check_condition(
                                   dataDictionary[field].min(), dataDictionary[field].max())




