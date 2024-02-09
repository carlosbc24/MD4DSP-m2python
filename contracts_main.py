
# Importing functions and classes from packages
from helpers.logger import set_logger
from tests.contract_pre_post.simple_test import ContractSimpleTest
from tests.contract_pre_post.tests_spotify_dataset import ContractExternalDatasetTests
from functions.contract_invariants import ContractsInvariants
import pandas as pd
from helpers.enumerations import DataType, DerivedType

if __name__ == "__main__":

    # Set the logger to save the logs of the execution in the path logs/test
    set_logger("test")

    # Execute all simple tests
    # contract_test = ContractSimpleTest()
    # contract_test.executeAll_SimpleTests()
    #
    # # Execute all external dataset tests
    # ContractTestWithDatasets = ContractExternalDatasetTests()
    # ContractTestWithDatasets.executeAll_ExternalDatasetTests()



    contracts = ContractsInvariants()

    # datadic= pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 2, 3, 4, 5], 'C': [1, 2, 3, 4, 5]})
    # result=contracts.checkInv_FixValue_FixValue(dataDictionary=datadic, dataTypeInput=DataType(2), FixValueInput=1,
    #                                   dataTypeOutput=DataType(6), FixValueOutput=3)
    #
    # print(datadic)
    # print(result)

    datadic=pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 3, 4, 5], 'C': [1, 2, 3, 4, 5]})
    result=contracts.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), FixValueInput=0,
                                      derivedTypeOutput=DerivedType(0), axis_param=0)

    print(datadic)
    print(result)
