import os

from helpers.logger import set_logger, print_and_log
from test.simple_test import ContractSimpleTest
from test.tests_spotify_dataset import ContractWithDatasetTests

if __name__ == "__main__":

    set_logger("test")

    contract_test = ContractSimpleTest()
    contract_test.executeAll_SimpleTests()

    ContractTestWithDatasets = ContractWithDatasetTests()
    # ContractTestWithDatasets.execute_CheckFieldRange_DatasetTests()
    # ContractTestWithDatasets.execute_CheckFixValueRangeString_DatasetTests()
    # ContractTestWithDatasets.execute_CheckFixValueRangeFloat_DatasetTests()
    # ContractTestWithDatasets.execute_CheckFixValueRangeDateTime_DatasetTests()
    # ContractTestWithDatasets.execute_CheckIntervalRangeFloat_DatasetTests()

    # ContractTestWithDatasets.executeAll_DatasetTests()




    ContractTestWithDatasets= ContractTestWithDatasets()
    #ContractTestWithDatasets.execute_CheckFieldRange_Tests()
    #ContractTestWithDatasets.execute_CheckFixValueRangeString_Tests()
    #ContractTestWithDatasets.execute_CheckFixValueRangeFloat_Tests()
    ContractTestWithDatasets.execute_CheckFixValueRangeDateTime_Tests()   #TODO: fix this test
    #ContractTestWithDatasets.execute_checkIntervalRangeFloat_Tests()