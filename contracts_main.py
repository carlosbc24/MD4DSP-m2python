# Importing functions and classes from packages
import time

import numpy as np
import pandas as pd

from helpers.auxiliar import format_duration
from helpers.logger import set_logger, print_and_log
from tests.contract_pre_post.simple_test import ContractSimpleTest
from tests.contract_pre_post.tests_spotify_dataset import ContractExternalDatasetTests
from tests.invariants.simple_test import InvariantSimpleTest
from tests.invariants.tests_spotify_dataset import InvariantsExternalDatasetTests



def execute_prepost_simple_tests():
    """
    Execute all pre-post simple tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all pre-post simple tests
    contract_test = ContractSimpleTest()
    contract_test.executeAll_SimpleTests()

    end = time.time()
    total_time = end - start

    print(f"Contract Simple Tests Duration: {format_duration(total_time)}")

def execute_prepost_external_dataset_tests():
    """
    Execute all pre-post external dataset tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all pre-post external dataset tests
    ContractTestWithDatasets = ContractExternalDatasetTests()
    ContractTestWithDatasets.executeAll_ExternalDatasetTests()

    end = time.time()
    total_time = end - start

    print(f"Contract External Dataset Tests Duration: {format_duration(total_time)}")

def execute_invariants_simple_tests():
    """
    Execute all invariants simple tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all invariants simple tests
    contracts_invariants_test = InvariantSimpleTest()
    contracts_invariants_test.execute_All_SimpleTests()

    end = time.time()
    total_time = end - start

    print(f"Invariants Simple Tests Duration: {format_duration(total_time)}")

def execute_invariants_external_dataset_tests():
    """
    Execute all invariants external dataset tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all invariants external dataset tests
    invariantTestWithDatasets = InvariantsExternalDatasetTests()
    # invariantTestWithDatasets.executeAll_ExternalDatasetTests()
    invariantTestWithDatasets.execute_WholeDatasetTests_checkInv_SpecialValue_NumOp_ExternalDataset()

    end = time.time()
    total_time = end - start

    print(f"Invariants External Dataset Tests Duration: {format_duration(total_time)}")

if __name__ == "__main__":
    # Set the logger to save the logs of the execution in the path logs/test
    set_logger("test")

    # Calculate execution time for each test block
    execute_prepost_simple_tests()
    # execute_prepost_external_dataset_tests()
    execute_invariants_simple_tests()
    execute_invariants_external_dataset_tests()

