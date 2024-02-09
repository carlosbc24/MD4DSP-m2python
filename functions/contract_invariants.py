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
    def checkInv_FixValue_FixValue(self, dataDictionary: pd.DataFrame, dataTypeInput: DataType, FixValueInput, dataTypeOutput: DataType, FixValueOutput) -> pd.DataFrame:
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
        FixValueInput, FixValueOutput=cast_type_FixValue(dataTypeInput, FixValueInput, dataTypeOutput, FixValueOutput)
        #Función auxiliar que cambia los valores de FixValueInput y FixValueOutput al tipo de dato en DataTypeInput y DataTypeOutput respectivamente
        dataDictionary = dataDictionary.replace(FixValueInput, FixValueOutput)
        return dataDictionary

    def checkInv_FixValue_DerivedValue(self, dataDictionary: pd.DataFrame, dataTypeInput: DataType, FixValueInput, derivedTypeOutput: DerivedType, axis_param: int=0) -> pd.DataFrame:
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
        FixValueInput, valorNulo=cast_type_FixValue(dataTypeInput, FixValueInput, None, None)
        #Función auxiliar que cambia el valor de FixValueInput al tipo de dato en DataTypeInput

        dataDictionary_copy = dataDictionary.copy()

        if derivedTypeOutput == DerivedType.MOSTFREQUENT:
            if axis_param == 1:
                dataDictionary_copy = dataDictionary_copy.apply(lambda fila: fila.apply(
                    lambda value: dataDictionary_copy.loc[
                        fila.name].value_counts().idxmax() if value == FixValueInput else value), axis=axis_param)
            elif axis_param == 0:
                dataDictionary_copy = dataDictionary_copy.apply(lambda columna: columna.apply(
                    lambda value: dataDictionary_copy[
                        columna.name].value_counts().idxmax() if value == FixValueInput else value), axis=axis_param)
        elif derivedTypeOutput == DerivedType.PREVIOUS:
            if axis_param == 1:
                #Hacer el previous para filas
                pass
            elif axis_param == 0:
                #Hacer el previous para columnas
                pass
        elif derivedTypeOutput == DerivedType.NEXT:
            if axis_param == 1:
                #Hacer el previous para filas
                pass
            elif axis_param == 0:
                #Hacer el previous para columnas
                pass


        return dataDictionary_copy







