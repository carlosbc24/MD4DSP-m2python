# Importing libraries
import numpy as np
import pandas as pd

from helpers.auxiliar import cast_type_FixValue
# Importing functions and classes from packages
from helpers.enumerations import Belong, Operator, Closure, DataType, DerivedType, Operation, SpecialType


class ContractsInvariants:
    # FixValue - FixValue, FixValue - DerivedValue, FixValue - NumOp
    # Interval - FixValue, Interval - DerivedValue, Interval - NumOp
    # SpecialValue - FixValue, SpecialValue - DerivedValue, SpecialValue - NumOp
    """

    """
    def checkInv_FixValue_FixValue(self, dataDictionary: pd.DataFrame, dataTypeInput: DataType, fixValueInput, dataTypeOutput: DataType, fixValueOutput) -> pd.DataFrame:
        """
        Check the invariant of the FixValue - FixValue relation
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            dataTypeOutput: data type of the output value
            FixValueOutput: output value to check
            axis_param: axis to check the invariant
        Returns:
            dataDictionary with the FixValueInput and FixValueOutput values changed to the type dataTypeInput and dataTypeOutput respectively
        """
        fixValueInput, fixValueOutput=cast_type_FixValue(dataTypeInput, fixValueInput, dataTypeOutput, fixValueOutput)
        #Función auxiliar que cambia los valores de FixValueInput y FixValueOutput al tipo de dato en DataTypeInput y DataTypeOutput respectivamente
        dataDictionary = dataDictionary.replace(fixValueInput, fixValueOutput)
        return dataDictionary

    def checkInv_FixValue_DerivedValue(self, dataDictionary: pd.DataFrame, dataTypeInput: DataType, fixValueInput, derivedTypeOutput: DerivedType, axis_param: int = None) -> pd.DataFrame:
        # Por defecto, si todos los valores son igual de frecuentes, se sustituye por el primer valor.
        # Comprobar si solo se debe hacer para filas y columnas o también para el dataframe completo.

        """
        Check the invariant of the FixValue - DerivedValue relation
        Sustituye el valor proporcionado por el usuario por el valor derivado en el eje que se especifique por parámetros
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            derivedTypeOutput: derived type of the output value
            axis_param: axis to check the invariant
        """
        fixValueInput, valorNulo=cast_type_FixValue(dataTypeInput, fixValueInput, None, None)
        #Función auxiliar que cambia el valor de FixValueInput al tipo de dato en DataTypeInput

        dataDictionary_copy = dataDictionary.copy()

        if derivedTypeOutput == DerivedType.MOSTFREQUENT:
            if axis_param == 1: # Aplica la función lambda a nivel de columna
                dataDictionary_copy = dataDictionary_copy.apply(lambda fila: fila.apply(
                    lambda value: dataDictionary_copy.loc[
                        fila.name].value_counts().idxmax() if value == fixValueInput else value), axis=axis_param)
            elif axis_param == 0: # Aplica la función lambda a nivel de fila
                dataDictionary_copy = dataDictionary_copy.apply(lambda columna: columna.apply(
                    lambda value: dataDictionary_copy[
                        columna.name].value_counts().idxmax() if value == fixValueInput else value), axis=axis_param)
            else: # Aplica la función lambda a nivel de dataframe
                # Asumiendo que 'dataDictionary_copy' es tu DataFrame y 'fixValueInput' el valor a reemplazar
                valor_mas_frecuente = dataDictionary_copy.stack().value_counts().idxmax()
                # Reemplaza 'fixValueInput' con el valor más frecuente en el DataFrame completo usando lambda
                dataDictionary_copy = dataDictionary_copy.apply(
                    lambda col: col.replace(fixValueInput, valor_mas_frecuente))

        elif derivedTypeOutput == DerivedType.PREVIOUS:
            # Aplica la función lambda a nivel de columna (axis=1) o a nivel de fila (axis=0)
            # Lambda que sustitutuye cualquier valor igual a FixValueInput del dataframe por el valor de la fila anterior en la misma columna
            dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput,
                                                                                  other=x.shift(1)), axis=axis_param)
        elif derivedTypeOutput == DerivedType.NEXT:
            # Aplica la función lambda a nivel de columna (axis=1) o a nivel de fila (axis=0)
            dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput,
                                                                                  other=x.shift(-1)), axis=axis_param)


        return dataDictionary_copy


    def checkInv_FixValue_NumOp(self, dataDictionary: pd.DataFrame, dataTypeInput: DataType, fixValueInput, numOpOutput: Operation, axis_param: int=0) -> pd.DataFrame:
        """
        Check the invariant of the FixValue - NumOp relation
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            numOpOutput: operation to check the invariant
            axis_param: axis to check the invariant
        Returns:
            dataDictionary with the FixValueInput values replaced by the result of the operation numOpOutput
        """
        fixValueInput, valorNulo=cast_type_FixValue(dataTypeInput, fixValueInput, None, None)
        #Función auxiliar que cambia el valor de FixValueInput al tipo de dato en DataTypeInput
        dataDictionary_copy = dataDictionary.copy()


        if numOpOutput == Operation.INTERPOLATION:
            # TODO: Revisar si se puede hacer la interpolación de todo el dataframe, ya que esto no funciona
            print("Not implemented yet")
            # dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput, other=x.interpolate()), axis=axis_param)
        elif numOpOutput == Operation.MEAN:
            if axis_param == None: # TODO: Revisar si se puede hacer la mediana de todo el dataframe, ya que esto no funciona
                print("Not implemented yet")
                # dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput, other=x.mean()))
            elif axis_param == 0 or axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput, other=x.mean()), axis=axis_param)
        elif numOpOutput == Operation.MEDIAN:
            if axis_param == None: # TODO: Revisar si se puede hacer la mediana de todo el dataframe, ya que esto no funciona
                print("Not implemented yet")
                # dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput, other=x.median()))
            elif axis_param == 0 or axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput, other=x.median()), axis=axis_param)
        elif numOpOutput == Operation.CLOSEST:
            # TODO: Revisar si se puede hacer la interpolación de todo el dataframe, ya que esto no funciona
            print("Not implemented yet")
            # dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput, other=x.interpolate(method='nearest')), axis=axis_param)
        else:
            raise ValueError("No valid operator")

        return dataDictionary_copy










