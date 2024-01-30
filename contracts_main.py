import os

from helpers.logger import set_logger
from test.simple_test import ContractTest
from test.tests_spotify_dataset import ContractTestWithDatasets

if __name__ == "__main__":

    set_logger("test")

    """contract_test = ContractTest()
    contract_test.execute_CheckFieldRange_Tests()
    contract_test.execute_CheckFixValueRangeString_Tests()
    contract_test.execute_CheckFixValueRangeFloat_Tests()
    contract_test.execute_CheckFixValueRangeDateTime_Tests()
    contract_test.execute_checkIntervalRangeFloat_Tests()"""

    ContractTestWithDatasets= ContractTestWithDatasets()
    ContractTestWithDatasets.execute_CheckFieldRange_Tests()
    #ContractTestWithDatasets.execute_CheckFixValueRangeString_Tests() #Este metodo falla y hay que implementarlo, ahora solo esta copiado del otro test. Los otros dos estan igual que este.
    ContractTestWithDatasets.execute_checkIntervalRangeFloat_Tests()