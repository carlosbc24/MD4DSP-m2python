
# Importing functions and classes from packages
from helpers.logger import set_logger
from test.simple_test import ContractSimpleTest
from test.tests_spotify_dataset import ContractWithDatasetTests

if __name__ == "__main__":

    set_logger("test")

    contract_test = ContractSimpleTest()
    contract_test.executeAll_SimpleTests()

    ContractTestWithDatasets = ContractWithDatasetTests()
    ContractTestWithDatasets.executeAll_DatasetTests()

