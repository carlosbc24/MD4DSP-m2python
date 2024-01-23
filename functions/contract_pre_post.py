import pandas as pd

from helpers.auxiliar import comparar_numeros
from helpers.enumerations import Belong, Operator


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
                        raise ValueError("Error: quant_rel and quant_abs should be None at the same time")
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
                            raise ValueError("Error: quant_rel and quant_abs should be None at the same time")
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
