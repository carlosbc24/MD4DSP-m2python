import os

from helpers.logger import set_logger, print_and_log
from test.simple_test import ContractSimpleTest
from test.tests_spotify_dataset import ContractWithDatasetTests

if __name__ == "__main__":

    set_logger("test")

    contract_test = ContractSimpleTest()
    # contract_test.executeAll_SimpleTests()

    ContractTestWithDatasets = ContractWithDatasetTests()
    ContractTestWithDatasets.executeAll_DatasetTests()
    # ContractTestWithDatasets.execute_CheckMissingRange_Tests()
    # ContractTestWithDatasets.execute_CheckInvalidValues_Tests()

