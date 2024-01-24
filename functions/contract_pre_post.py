from datetime import datetime

import numpy as np
import pandas as pd

from helpers.auxiliar import comparar_numeros
from helpers.enumerations import Belong, Operator, Closure


class ContractsPrePost:
    def checkFieldRange(self, fields: list, dataDictionary: pd.DataFrame, belongOp: Belong) -> bool:
        """
        Check if fields meets the condition of belongOp in dataDictionary.
        If belongOp is Belong.BELONG, then it checks if all fields are in dataDictionary.
        If belongOp is Belong.NOTBELONG, then it checks if any field in 'fields' are not in dataDictionary.

        :param fields: list of columns
        :param:dataDictionary: data dictionary
        :param:belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG

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

        :param value: value to check
        :param:dataDictionary: data dictionary
        :param:belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param:field: dataset column in which value will be checked
        :param:quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
               Operator.LESS=3, Operator.EQUAL=4
        :param:quant_abs: ?
        :param:quant_rel: ?

        :return: if fields meets the condition of belongOp in dataDictionary and field
        :rtype: bool
        """
        """
        Si field es None: 
            Si belongOp es BELONG 
                Si quant_op es None: comprueba que el valor especificado está en el dataset 
                Si no:  
                    Si quant_rel no es None: comprueba que el valor especificado está en el dataset y que además cumple lo que define el operador Operator sobre la cantidad relativa de veces que aparece. 
                    Si no, si quant_abs no es None: comprueba que el valor especificado está en el dataset y que además cumple lo que define el operador Operator sobre la cantidad absoluta de veces que aparece. 
                    Si no: error 
            Si belongOp es NOTBELONG y quant_op, quant_rel y quant_abs son None: comprueba que el valor especificado no está en el dataset 
            Si no: error 
        Si field no es None:  
            Si belongOp es BELONG 
                Si quant_op es None: comprueba que el valor especificado está en la columna especificada en field 
                Si no:  
                    Si quant_rel no es None: comprueba que el valor especificado está en la columna especificada en field y que además cumple lo que define el operador Operator sobre la cantidad relativa de veces que aparece. 
                    Si no, si quant_abs no es None: comprueba que el valor especificado está en la columna especificada en field y que además cumple lo que define el operador Operator sobre la cantidad absoluta de veces que aparece. 
                    Si no: error 
            Si belongOp es NOTBELONG y quant_op, quant_rel y quant_abs son None: comprueba que el valor especificado no está en la columna especificada en field  
            Si no: error 
        """
        dataDictionary = dataDictionary.replace({np.nan: None})
        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if value is in dataDictionary
                    if value in dataDictionary.values:
                        return True
                    return False
                else:
                    if quant_rel is not None and quant_abs is None:  # Check if value is in dataDictionary and if it meets the condition of quant_rel
                        if value in dataDictionary.values and comparar_numeros(dataDictionary.value_counts(dropna=False).get(value, 0)
                                                    / dataDictionary.size, quant_rel, quant_op):#Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                            # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                            return True
                        return False
                    elif quant_rel is not None and quant_abs is not None:
                        # Añadido respecto a la especificacion inicial del contrato para test case 4
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError("Error: quant_rel and quant_abs can't have different values than None at the same time")
                    elif quant_abs is not None:
                        if value in dataDictionary.values and comparar_numeros(dataDictionary.value_counts(dropna=False).get(value, 0),
                                                                               quant_abs, quant_op):#Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                            return True
                        return False
                    else:
                        raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    if value not in dataDictionary.values:
                        return True
                    return False
                else:
                    raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")
        else:
            if field is not None:
                if belongOp == Belong.BELONG:
                    if quant_op is None:
                        if value in dataDictionary[field].values:
                            return True
                        return False
                    else:
                        if quant_rel is not None and quant_abs is None: # Añadido respecto a la especificacion inicial del contrato para test case 4
                            if value in dataDictionary[field].values and comparar_numeros(dataDictionary[field].value_counts(dropna=False).get(value, 0)
                                                    / dataDictionary.size, quant_rel, quant_op):
                                # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                                return True
                            return False
                        elif quant_rel is not None and quant_abs is not None:
                            # Añadido respecto a la especificacion inicial del contrato para test case 4
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError("Error: quant_rel and quant_abs can't have different values than None at the same time")
                        elif quant_abs is not None:
                            if value in dataDictionary[field].values and comparar_numeros(dataDictionary[field].value_counts(dropna=False).get(value, 0),
                                                                                          quant_abs, quant_op):
                                return True
                            return False
                        else: # quant_rel is None and quant_abs is None
                            raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        if value not in dataDictionary[field].values:
                            return True
                        return False
                    else:
                        raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")




    def checkFixValueRangeFloat(self, value: float, dataDictionary: pd.DataFrame, belongOp: Belong, field: str = None,
                                 quant_abs: int = None, quant_rel: float = None, quant_op: Operator = None) -> bool:
        """
        Check if fields meets the condition of belongOp in dataDictionary

        :param value: value to check
        :param:dataDictionary: data dictionary
        :param:belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param:field: dataset column in which value will be checked
        :param:quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
               Operator.LESS=3, Operator.EQUAL=4
        :param:quant_abs: ?
        :param:quant_rel: ?

        :return: if fields meets the condition of belongOp in dataDictionary and field
        :rtype: bool
        """
        """
        Si field es None: 
            Si belongOp es BELONG 
                Si quant_op es None: comprueba que el valor especificado está en el dataset 
                Si no:  
                    Si quant_rel no es None: comprueba que el valor especificado está en el dataset y que además cumple lo que define el operador Operator sobre la cantidad relativa de veces que aparece. 
                    Si no, si quant_abs no es None: comprueba que el valor especificado está en el dataset y que además cumple lo que define el operador Operator sobre la cantidad absoluta de veces que aparece. 
                    Si no: error 
            Si belongOp es NOTBELONG y quant_op, quant_rel y quant_abs son None: comprueba que el valor especificado no está en el dataset 
            Si no: error 
        Si field no es None:  
            Si belongOp es BELONG 
                Si quant_op es None: comprueba que el valor especificado está en la columna especificada en field 
                Si no:  
                    Si quant_rel no es None: comprueba que el valor especificado está en la columna especificada en field y que además cumple lo que define el operador Operator sobre la cantidad relativa de veces que aparece. 
                    Si no, si quant_abs no es None: comprueba que el valor especificado está en la columna especificada en field y que además cumple lo que define el operador Operator sobre la cantidad absoluta de veces que aparece. 
                    Si no: error 
            Si belongOp es NOTBELONG y quant_op, quant_rel y quant_abs son None: comprueba que el valor especificado no está en la columna especificada en field  
            Si no: error 
        """
        dataDictionary = dataDictionary.replace({np.nan: None})     #Se sustituyen los NaN por None para que no de error al hacer la comparacion de None con NaN. Como el dataframe es de floats, los None se convierten en NaN
        if value is not None:       #Se castea el valor a float para que no de error al hacer un get por valor, porque al hacer un get detecta el valor como int
            value = float(value)

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if value is in dataDictionary
                    if value in dataDictionary.values:
                        return True
                    return False
                else:
                    if quant_rel is not None and quant_abs is None:  # Check if value is in dataDictionary and if it meets the condition of quant_rel
                        if value in dataDictionary.values and comparar_numeros(dataDictionary.value_counts(dropna=False).get(value, 0)
                                                    / dataDictionary.size, quant_rel, quant_op):#Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                            # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                            return True
                        return False
                    elif quant_rel is not None and quant_abs is not None:
                        # Añadido respecto a la especificacion inicial del contrato para test case 4
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError("quant_rel and quant_abs can't have different values than None at the same time")
                    elif quant_abs is not None:
                        if value in dataDictionary.values and comparar_numeros(dataDictionary.value_counts(dropna=False).get(value, 0),
                                                                               quant_abs, quant_op):#Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                            return True
                        return False
                    else:
                        raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    if value not in dataDictionary.values:
                        return True
                    return False
                else:
                    raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")
        else:
            if field is not None:
                if belongOp == Belong.BELONG:
                    if quant_op is None:
                        if value in dataDictionary[field].values:
                            return True
                        return False
                    else:
                        if quant_rel is not None and quant_abs is None: # Añadido respecto a la especificacion inicial del contrato para test case 4
                            if value in dataDictionary[field].values and comparar_numeros(dataDictionary[field].value_counts(dropna=False).get(value, 0)
                                                    / dataDictionary.size, quant_rel, quant_op):
                                # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                                return True
                            return False
                        elif quant_rel is not None and quant_abs is not None:
                            # Añadido respecto a la especificacion inicial del contrato para test case 4
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError("quant_rel and quant_abs can't have different values than None at the same time")
                        elif quant_abs is not None:
                            if value in dataDictionary[field].values and comparar_numeros(dataDictionary[field].value_counts(dropna=False).get(value, 0),
                                                                                          quant_abs, quant_op):
                                return True
                            return False
                        else: # quant_rel is None and quant_abs is None
                            raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        if value not in dataDictionary[field].values:
                            return True
                        return False
                    else:
                        raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")




    def checkFixValueRangeDateTime(self, value: datetime, dataDictionary: pd.DataFrame, belongOp: Belong, field: str = None,
                                 quant_abs: int = None, quant_rel: float = None, quant_op: Operator = None) -> bool:
        """
        Check if fields meets the condition of belongOp in dataDictionary

        :param value: value to check
        :param:dataDictionary: data dictionary
        :param:belongOp: enum operator which can be Belong.BELONG or Belong.NOTBELONG
        :param:field: dataset column in which value will be checked
        :param:quant_op: enum operator which can be Operator.GREATEREQUAL=0, Operator.GREATER=1, Operator.LESSEQUAL=2,
               Operator.LESS=3, Operator.EQUAL=4
        :param:quant_abs: ?
        :param:quant_rel: ?

        :return: if fields meets the condition of belongOp in dataDictionary and field
        :rtype: bool
        """
        """
        Si field es None: 
            Si belongOp es BELONG 
                Si quant_op es None: comprueba que el valor especificado está en el dataset 
                Si no:  
                    Si quant_rel no es None: comprueba que el valor especificado está en el dataset y que además cumple lo que define el operador Operator sobre la cantidad relativa de veces que aparece. 
                    Si no, si quant_abs no es None: comprueba que el valor especificado está en el dataset y que además cumple lo que define el operador Operator sobre la cantidad absoluta de veces que aparece. 
                    Si no: error 
            Si belongOp es NOTBELONG y quant_op, quant_rel y quant_abs son None: comprueba que el valor especificado no está en el dataset 
            Si no: error 
        Si field no es None:  
            Si belongOp es BELONG 
                Si quant_op es None: comprueba que el valor especificado está en la columna especificada en field 
                Si no:  
                    Si quant_rel no es None: comprueba que el valor especificado está en la columna especificada en field y que además cumple lo que define el operador Operator sobre la cantidad relativa de veces que aparece. 
                    Si no, si quant_abs no es None: comprueba que el valor especificado está en la columna especificada en field y que además cumple lo que define el operador Operator sobre la cantidad absoluta de veces que aparece. 
                    Si no: error 
            Si belongOp es NOTBELONG y quant_op, quant_rel y quant_abs son None: comprueba que el valor especificado no está en la columna especificada en field  
            Si no: error 
        """

        if field is None:
            if belongOp == Belong.BELONG:
                if quant_op is None:  # Check if value is in dataDictionary
                    if value in dataDictionary.values:
                        return True
                    return False
                else:
                    if quant_rel is not None and quant_abs is None:  # Check if value is in dataDictionary and if it meets the condition of quant_rel
                        if value in dataDictionary.values and comparar_numeros(dataDictionary.value_counts(dropna=False).get(value, 0)
                                                    / dataDictionary.size, quant_rel, quant_op):#Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                            # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                            return True
                        return False
                    elif quant_rel is not None and quant_abs is not None:
                        # Añadido respecto a la especificacion inicial del contrato para test case 4
                        # Si se proporcionan los dos, se lanza un ValueError
                        raise ValueError("quant_rel and quant_abs can't have different values than None at the same time")
                    elif quant_abs is not None:
                        if value in dataDictionary.values and comparar_numeros(dataDictionary.value_counts(dropna=False).get(value, 0),
                                                                               quant_abs, quant_op):#Si field es None, en lugar de buscar en una columna, busca en el dataframe completo
                            return True
                        return False
                    else:
                        raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
            else:
                if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                    if value not in dataDictionary.values:
                        return True
                    return False
                else:
                    raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")
        else:
            if field is not None:
                if belongOp == Belong.BELONG:
                    if quant_op is None:
                        if value in dataDictionary[field].values:
                            return True
                        return False
                    else:
                        if quant_rel is not None and quant_abs is None: # Añadido respecto a la especificacion inicial del contrato para test case 4
                            if value in dataDictionary[field].values and comparar_numeros(dataDictionary[field].value_counts(dropna=False).get(value, 0)
                                                    / dataDictionary.size, quant_rel, quant_op):
                                # Importante el dropna=False para que cuente los valores NaN en caso de que value sea None
                                return True
                            return False
                        elif quant_rel is not None and quant_abs is not None:
                            # Añadido respecto a la especificacion inicial del contrato para test case 4
                            # Si se proporcionan los dos, se lanza un ValueError
                            raise ValueError("quant_rel and quant_abs can't have different values than None at the same time")
                        elif quant_abs is not None:
                            if value in dataDictionary[field].values and comparar_numeros(dataDictionary[field].value_counts(dropna=False).get(value, 0),
                                                                                          quant_abs, quant_op):
                                return True
                            return False
                        else: # quant_rel is None and quant_abs is None
                            raise ValueError("Error: quant_rel or quant_abs should be provided when belongOp is BELONG and quant_op is not None")
                else:
                    if belongOp == Belong.NOTBELONG and quant_op is None and quant_rel is None and quant_abs is None:
                        if value not in dataDictionary[field].values:
                            return True
                        return False
                    else:
                        raise ValueError("Error: quant_rel and quant_abs should be None when belongOp is NOTBELONG")


    def checkIntervalRangeFloat(leftMargin:float, rightMargin:float, dataDictionary: pd.DataFrame, closureType:Closure, belongOp:Belong,  field:str=None)->bool:
        """
        :param rightMargin:
        :param dataDictionary:
        :param closureType:
        :param belongOp:
        :param field:
        :return:
        """
        """
        Si field es None: 
            Comprueba si el rango establecido a través de los valores rightMargin y leftMargin y el operador closureType cumplen con la condición definida por el operador BelongOp en el dataset de entrada. 
        Si no: 
            Comprueba si el rango establecido a través de los valores rightMargin y leftMargin y el operador closureType cumplen con la condición definida por el operador BelongOp en la columna especificada. 
        """
        #TODO: Comprobar que el rango es correcto (leftMargin <= rightMargin)
        #Se ha supuesto que los valores del dataset tienen que pertenecer al rango (o no) en función de belongOp y closureType
        if field is None:
            if belongOp == Belong.BELONG:
                if closureType == Closure.openOpen:
                    if leftMargin < dataDictionary.min() and rightMargin > dataDictionary.max():
                        return True
                    return False
                elif closureType == Closure.openClosed:
                    if leftMargin< dataDictionary.min() and rightMargin >= dataDictionary.max():
                        return True
                    return False
                elif closureType == Closure.closedOpen:
                    if leftMargin <= dataDictionary.min() and rightMargin > dataDictionary.max():
                        return True
                    return False
                elif closureType == Closure.closedClosed:
                    if leftMargin <= dataDictionary.min() and rightMargin >= dataDictionary.max():
                        return True
                    return False
                else:
                    raise ValueError("Error: closureType should be openOpen, openClosed, closedOpen or closedClosed")
            else:
                if belongOp == Belong.NOTBELONG:
                    if closureType == Closure.openOpen:
                        if leftMargin < dataDictionary.min() and rightMargin > dataDictionary.max():
                            return False
                        return True
                    elif closureType == Closure.openClosed:
                        if leftMargin< dataDictionary.min() and rightMargin >= dataDictionary.max():
                            return False
                        return True
                    elif closureType == Closure.closedOpen:
                        if leftMargin <= dataDictionary.min() and rightMargin > dataDictionary.max():
                            return False
                        return True
                    elif closureType == Closure.closedClosed:
                        if leftMargin <= dataDictionary.min() and rightMargin >= dataDictionary.max():
                            return False
                        return True
                    else:
                        raise ValueError("Error: closureType should be openOpen, openClosed, closedOpen or closedClosed")
        else:
            if belongOp == Belong.BELONG:
                if closureType == Closure.openOpen:
                    if leftMargin < dataDictionary[field].min() and rightMargin > dataDictionary[field].max():
                        return True
                    return False
                elif closureType == Closure.openClosed:
                    if leftMargin < dataDictionary[field].min() and rightMargin >= dataDictionary[field].max():
                        return True
                    return False
                elif closureType == Closure.closedOpen:
                    if leftMargin <= dataDictionary[field].min() and rightMargin > dataDictionary[field].max():
                        return True
                    return False
                elif closureType == Closure.closedClosed:
                    if leftMargin <= dataDictionary[field].min() and rightMargin >= dataDictionary[field].max():
                        return True
                    return False
                else:
                    raise ValueError("Error: closureType should be openOpen, openClosed, closedOpen or closedClosed")
            else:
                if belongOp == Belong.NOTBELONG:
                    if closureType == Closure.openOpen:
                        if leftMargin < dataDictionary[field].min() and rightMargin > dataDictionary[field].max():
                            return False
                        return True
                    elif closureType == Closure.openClosed:
                        if leftMargin< dataDictionary[field].min() and rightMargin >= dataDictionary[field].max():
                            return False
                        return True
                    elif closureType == Closure.closedOpen:
                        if leftMargin <= dataDictionary[field].min() and rightMargin > dataDictionary[field].max():
                            return False
                        return True
                    elif closureType == Closure.closedClosed:
                        if leftMargin <= dataDictionary[field].min() and rightMargin >= dataDictionary[field].max():
                            return False
                        return True
                    else:
                        raise ValueError("Error: closureType should be openOpen, openClosed, closedOpen or closedClosed")









