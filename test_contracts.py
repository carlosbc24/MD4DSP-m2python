# Importing functions and classes from packages
import time

from helpers.auxiliar import format_duration
from helpers.logger import set_logger
from tests.contract_invariants.simple_test import InvariantsSimpleTest
from tests.contract_invariants.tests_spotify_dataset import InvariantsExternalDatasetTests
from tests.contract_pre_post.simple_test import PrePostSimpleTest
from tests.contract_pre_post.tests_spotify_dataset import ContractExternalDatasetTests


def execute_prepost_simple_tests():
    """
    Execute all pre-post simple tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all pre-post simple tests
    prepost_simple_tests = PrePostSimpleTest()
    prepost_simple_tests.executeAll_SimpleTests()

    end = time.time()
    total_time = end - start

    print(f"Contract Simple Tests Duration: {format_duration(total_time)}")


def execute_prepost_external_dataset_tests():
    """
    Execute all pre-post external dataset tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all pre-post external dataset tests
    prepost_tests_with_external_dataset = ContractExternalDatasetTests()
    prepost_tests_with_external_dataset.executeAll_ExternalDatasetTests()

    end = time.time()
    total_time = end - start

    print(f"Contract External Dataset Tests Duration: {format_duration(total_time)}")


def execute_invariants_simple_tests():
    """
    Execute all invariants simple tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all invariants simple tests
    invariant_simple_tests = InvariantsSimpleTest()
    invariant_simple_tests.execute_All_SimpleTests()

    end = time.time()
    total_time = end - start

    print(f"Invariants Simple Tests Duration: {format_duration(total_time)}")


def execute_invariants_external_dataset_tests():
    """
    Execute all invariants external dataset tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all invariants external dataset tests
    invariant_tests_with_external_dataset = InvariantsExternalDatasetTests()
    invariant_tests_with_external_dataset.executeAll_ExternalDatasetTests()

    end = time.time()
    total_time = end - start

    print(f"Invariants External Dataset Tests Duration: {format_duration(total_time)}")


if __name__ == "__main__":
    # Set the logger to save the logs of the execution in the path logs/test
    set_logger("test_contracts")

    # FINAL TESTS
    # Calculate execution time for each test block (pre-post)
    # execute_prepost_sitivate al_dataset_tests()

    # Calculate execution time for each test block (invariants)
    execute_invariants_simple_tests()
    execute_invariants_external_dataset_tests()
