# Importing libraries
import unittest
import numpy as np
import pandas as pd
from tqdm import tqdm

# Importing functions and classes from packages
from functions.contract_invariants import ContractsInvariants
from helpers.enumerations import Belong, Operator, Closure, DataType, DerivedType, SpecialType, Operation
from helpers.logger import print_and_log


class InvariantSimpleTest(unittest.TestCase):
    """
    Class to test the contracts with simple test cases

    Attributes:
    pre_post (InvariantSimpleTest): instance of the class InvariantSimpleTest

    Methods:
    """

    def __init__(self):
        """
        Constructor of the class

        Attributes:
        inv (InvariantSimpleTest): instance of the class InvariantSimpleTest
        """
        self.invs = ContractsInvariants()


    def test_invs(self):
        """
        # datadic= pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 2, 3, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # result=contracts.checkInv_FixValue_FixValue(dataDictionary=datadic, dataTypeInput=DataType(2), FixValueInput=1,
        #                                   dataTypeOutput=DataType(6), FixValueOutput=3)
        #
        # print(datadic)
        # print(result)

        # datadic=pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # result=contracts.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), fixValueInput=0,
        #                                                 derivedTypeOutput=DerivedType(0), axis_param=None)

        # datadic=pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # print(datadic)
        # result = contracts.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2),
        #                                                   fixValueInput=5,
        #                                                   derivedTypeOutput=DerivedType(1), axis_param=0)
        # print(result)
        #
        # datadic=pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # print(datadic)
        # result = contracts.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2),
        #                                                   fixValueInput=2,
        #                                                   derivedTypeOutput=DerivedType(2), axis_param=1)
        # print(result)

        # datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # print(datadic)
        # result = contracts.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
        #                                                   fixValueInput=0, numOpOutput=Operation(0), axis_param=0)
        # print(result)

        # datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # print(datadic)
        # result = contracts.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
        #                                                   fixValueInput=0, numOpOutput=Operation(3), axis_param=1)
        # print(result)

        # datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # print(datadic)
        # result = contracts.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
        #                                               closureType=Closure(2), dataTypeOutput=DataType(0), fixValueOutput='Suspenso')
        # print(result)

        # datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # print(datadic)
        # result = contracts.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5, closureType=Closure(1),
        #                                                   derivedTypeOutput=DerivedType(0), axis_param=None)
        # print(result)

        # datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # print(datadic)
        # result = contracts.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5, closureType=Closure(1),
        #                                                   derivedTypeOutput=DerivedType(1), axis_param=1)
        # print(result)

        # datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # print(datadic)
        # result = contracts.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
        #                                            closureType=Closure(1), numOpOutput=Operation(0), axis_param=None)
        # print(result)

        # datadic = pd.DataFrame({'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10], 'E': [1, 10, 3, 4, 1]})
        # missing_values=[1,3,4]
        # print(datadic)
        # result = contracts.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
        #                                            dataTypeOutput=DataType(2), fixValueOutput=999,
        #                                             missing_values=missing_values, axis_param=1)
        # print(result)

        # datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8, 6, 1, 2]})
        # missing_values=[1,3,4]
        # print_and_log(datadic)
        # print(datadic)
        # result = contracts.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
        #                                                     derivedTypeOutput=DerivedType(0), missing_values=missing_values,
        #                                                       axis_param=0)
        # print_and_log(result)
        # print(result)
        """

    def execute_checkInv_SpecialValue_NumOp_SimpleTests(self):
        """
        Method to test the checkInv_SpecialValue_NumOp method with simple test cases
        """
        print_and_log("Testing checkInv_SpecialValue_NumOp method with simple test cases")
        """
        SpecialTypes:
            0: Missing
            1: Invalid
            2: Outlier
        Operation:
            0: Interpolation
            1: Mean
            2: Median
            3: Closest
        Axis:
            0: Columns
            1: Rows
            None: All
        """


        #Caso 1
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8, 6, 1, 2]})
        missing_values = [1, 3, 4]
        manual = pd.DataFrame({'A': [0, 2, 3, 2, 2], 'B': [2, 3, 4.5, 6, 12], 'C': [10, 6.5, 2, 1.5, 0], 'D': [5, 8, 6, 4, 2]})
        result = self.invs.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                        numOpOutput=Operation(0), missing_values=missing_values,
                                                                        axis_param=0)

        print_and_log(datadic)
        print(datadic)

        #pd.testing.assert_frame_equal(manual, result)

        print_and_log(result)
        print(result)

