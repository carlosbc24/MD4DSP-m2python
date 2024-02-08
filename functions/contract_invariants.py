# Importing libraries
import numpy as np
import pandas as pd

# Importing functions and classes from packages
from helpers.enumerations import Belong, Operator, Closure, DataType, DerivedType, Operation, SpecialType


class ContractsInvariants:
    """
    This class is responsible for checking invariants of contracts.
    """

    def check_FixValue_FixValue(self, dataDictionary: pd.DataFrame, dataType: DataType, fixValue) -> bool:
        # Check FixValue - FixValue
        dataDictionary.apply(lambda func: func*2 if(func.iloc[0]==2) else func, axis=1)
        print()
