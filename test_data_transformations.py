# Importing functions and classes from packages
import time
from helpers.logger import set_logger
from helpers.auxiliar import format_duration
from tests.data_transformations.simple_test import DataTransformationsSimpleTest
from tests.data_transformations.tests_spotify_dataset import DataTransformationsExternalDatasetTests


def execute_data_transformations_simple_tests():
    """
    Execute all data_transformations simple tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all data_transformations simple tests
    data_transformation_simple_tests = DataTransformationsSimpleTest()
    data_transformation_simple_tests.execute_All_SimpleTests()

    end = time.time()
    total_time = end - start

    print(f"Data Transformations Simple Tests Duration: {format_duration(total_time)}")


def execute_data_transformations_external_dataset_tests():
    """
    Execute all data_transformations external dataset tests and calculate the duration of the execution.
    """
    start = time.time()

    # Execute all data_transformations external dataset tests
    data_transformations_tests_with_external_dataset = DataTransformationsExternalDatasetTests()
    data_transformations_tests_with_external_dataset.executeAll_ExternalDatasetTests()

    end = time.time()
    total_time = end - start

    print(f"Data Transformations External Dataset Tests Duration: {format_duration(total_time)}")


if __name__ == "__main__":
    # Set the logger to save the logs of the execution in the path logs/test
    set_logger("test_data_transformations")

    # FINAL TESTS
    # Calculate execution time for each test block (data_transformations)
    execute_data_transformations_simple_tests()
    execute_data_transformations_external_dataset_tests()
