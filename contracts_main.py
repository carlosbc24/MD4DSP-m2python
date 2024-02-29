
# Importing functions and classes from packages
from helpers.logger import set_logger
from tests.contract_pre_post.simple_test import ContractSimpleTest
from tests.contract_pre_post.tests_spotify_dataset import ContractExternalDatasetTests
from tests.invariants.simple_test import InvariantSimpleTest
from tests.invariants.tests_spotify_dataset import InvariantsExternalDatasetTests

if __name__ == "__main__":

    # Set the logger to save the logs of the execution in the path logs/test
    set_logger("test")

    # Execute all pre-post simple tests
    # contract_test = ContractSimpleTest()
    # contract_test.executeAll_SimpleTests()
    # contract_test.execute_CheckOutliers_SimpleTests()

    # Execute all pre-post external dataset tests
    # ContractTestWithDatasets = ContractExternalDatasetTests()
    # ContractTestWithDatasets.executeAll_ExternalDatasetTests()
    # ContractTestWithDatasets.execute_CheckOutliers_SpotifyDatasetTests()

    # Execute all invariants simple tests
    contracts_invariants_test = InvariantSimpleTest()
    contracts_invariants_test.execute_All_SimpleTests()

    # TODO: Implement the invariants external dataset tests
    # Execute all invariants external dataset tests
    # InvariantTestWithDatasets = InvariantsExternalDatasetTests()
    # InvariantTestWithDatasets.executeAll_ExternalDatasetTests()


