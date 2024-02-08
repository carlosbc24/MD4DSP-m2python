# Importing libraries
import numpy as np
import pandas as pd

from helpers.auxiliar import cast_type_FixValue
# Importing functions and classes from packages
from helpers.enumerations import Belong, Operator, Closure, DataType, DerivedType, Operation, SpecialType


class ContractsInvariants:
    """
    This class is responsible for checking invariants of contracts.
    """
    def checkInv_FixValue_FixValue(self, dataDictionary: pd.DataFrame, dataTypeInput: DataType, FixValueInput, dataTypeOutput: DataType, FixValueOutput, axis_param: int=0) -> pd.DataFrame:
        FixValueInput, FixValueOutput=cast_type_FixValue(dataTypeInput, FixValueInput, dataTypeOutput, FixValueOutput)
        #Función auxiliar que cambia los valores de FixValueInput y FixValueOutput al tipo de dato en DataTypeInput y DataTypeOutput respectivamente
        # Check FixValue - FixValue
        dataDictionary=dataDictionary.apply(lambda func: FixValueOutput if(func==FixValueInput) else func, axis=axis_param)
        return dataDictionary

