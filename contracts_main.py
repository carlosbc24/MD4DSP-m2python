import os

from helpers.logger import set_logger, print_and_log
from test.simple_test import ContractSimpleTest
from test.tests_spotify_dataset import ContractWithDatasetTests

if __name__ == "__main__":

    set_logger("test")

    contract_test = ContractSimpleTest()
    contract_test.execute_CheckFieldRange_Tests()
    contract_test.execute_CheckFixValueRangeString_Tests()
    contract_test.execute_CheckFixValueRangeFloat_Tests()
    contract_test.execute_CheckFixValueRangeDateTime_Tests()
    contract_test.execute_checkIntervalRangeFloat_Tests()

    ContractTestWithDatasets= ContractWithDatasetTests()
    ContractTestWithDatasets.execute_CheckFieldRange_Tests()
    # ContractTestWithDatasets.execute_CheckFixValueRangeString_Tests()
    # ContractTestWithDatasets.execute_CheckFixValueRangeFloat_Tests()
    # ContractTestWithDatasets.execute_CheckFixValueRangeDateTime_Tests()
    # ContractTestWithDatasets.execute_checkIntervalRangeFloat_Tests()
