# Importing functions and classes from packages
import time

from helpers.auxiliar import format_duration
from helpers.logger import set_logger

from tests.data_smells.simple_test import DataSmellsSimpleTest
from tests.data_smells.tests_spotify_dataset import DataSmellExternalDatasetTests


def execute_data_smells_simple_tests():
    """
    Execute all data smells simple tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all data smells simple tests
    data_smells_simple_tests = DataSmellsSimpleTest()
    data_smells_simple_tests.executeAll_SimpleTests()

    end = time.time()
    total_time = end - start

    print(f"Data Smells Simple Tests Duration: {format_duration(total_time)}")


def execute_data_smells_external_dataset_tests():
    """
    Execute all data smells external dataset tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all data smells external dataset tests
    data_smells_tests_with_external_dataset = DataSmellExternalDatasetTests()
    data_smells_tests_with_external_dataset.executeAll_ExternalDatasetTests()

    end = time.time()
    total_time = end - start

    print(f"Data Smells External Dataset Tests Duration: {format_duration(total_time)}")


if __name__ == "__main__":
    # Set the logger to save the logs of the execution in the path logs/test
    set_logger("test_data_smells")

    # Execute all tests and calculate execution time
    execute_data_smells_simple_tests()
    execute_data_smells_external_dataset_tests()
