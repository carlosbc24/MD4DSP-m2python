
# Importing functions and classes from packages
from helpers.logger import set_logger
from tests.contract_pre_post.simple_test import ContractSimpleTest
from tests.contract_pre_post.tests_spotify_dataset import ContractExternalDatasetTests

if __name__ == "__main__":

    # Set the logger to save the logs of the execution in the path logs/test
    set_logger("test")

    # Execute all simple tests
    contract_test = ContractSimpleTest()
    contract_test.executeAll_SimpleTests()

    # Execute all external dataset tests
    ContractTestWithDatasets = ContractExternalDatasetTests()
    ContractTestWithDatasets.executeAll_ExternalDatasetTests()

