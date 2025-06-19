# Importing libraries
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm
import functions.data_smells as data_smells

# Importing functions and classes from packages
from helpers.logger import print_and_log


class DataSmellsSimpleTest(unittest.TestCase):
    """
            Class to test the Data Smells Simple Tests

        Attributes:
            data_smells (DataSmells): instance of the class DataSmells

        Methods:
            execute_All_SimpleTests: execute all the simple tests of the functions of the class
        """
    def __init__(self):
        """
        Constructor of the class

        Attributes:
            data_smells (DataSmells): instance of the class DataSmells

        Functions:
            executeAll_SimpleTests: execute all the simple tests of the data smells functions of the class
        """
        super().__init__()
        self.data_smells = data_smells

    def executeAll_SimpleTests(self):
        """
        Execute all the simple tests of the functions of the class
        """
        simple_test_methods = [
            self.execute_check_precision_consistency_SimpleTests,
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ STARTING DATA-SMELL SIMPLE TEST CASES -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for simple_test_method in tqdm(simple_test_methods, desc="Running Data Smell Simple Tests",
                                       unit="test"):
            simple_test_method()

        print_and_log("")
        print_and_log("-----------------------------------------------------")
        print_and_log("-- DATA-SMELL SIMPLE TEST CASES EXECUTION FINISHED --")
        print_and_log("-----------------------------------------------------")
        print_and_log("")

    def execute_check_precision_consistency_SimpleTests(self):
        """
        Execute simple tests for check_precision_consistency function.
        Tests the following cases:
        1. Invalid expected_decimals parameter (negative)
        2. Invalid expected_decimals parameter (non-integer)
        3. Non-existent field
        4. Non-numeric field
        5. Inconsistent decimal places (multiple lengths)
        6. Fixed but incorrect number of decimals
        7. Correct number of decimals (success case)
        8. Integer field (should pass with 0 decimals)
        9. Check all numeric fields at once
        10. Empty DataFrame
        11. Column with all NaN values
        12. Mixed integer and float values
        """
        print_and_log("")
        print_and_log("Testing check_precision_consistency function...")

        # Create test data
        data = {
            'numeric_consistent': [1.234, 5.678, 9.012],
            'numeric_inconsistent': [1.2, 3.45, 6.789],
            'numeric_wrong_decimals': [1.23, 4.56, 7.89],
            'non_numeric': ['a', 'b', 'c'],
            'integer_field': [1, 2, 3],
            'all_nan': [np.nan, np.nan, np.nan],
            'mixed_types': [1, 2.5, 3.0]
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test Case 1: Negative decimals
        expected_exception = TypeError
        with self.assertRaises(expected_exception):
            self.data_smells.check_precision_consistency(df, -1, 'numeric_consistent')
        print_and_log("Test Case 1 Passed: Expected TypeError, got TypeError")

        # Test Case 2: Non-integer decimals
        expected_exception = TypeError
        with self.assertRaises(expected_exception):
            self.data_smells.check_precision_consistency(df, 2.5, 'numeric_consistent')
        print_and_log("Test Case 2 Passed: Expected TypeError, got TypeError")

        # Test Case 3: Non-existent field
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_smells.check_precision_consistency(df, 3, 'non_existent_field')
        print_and_log("Test Case 3 Passed: Expected ValueError, got ValueError")

        # Test Case 4: Non-numeric field
        result = self.data_smells.check_precision_consistency(df, 3, 'non_numeric')
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Test Case 5: Inconsistent decimal places
        result = self.data_smells.check_precision_consistency(df, 3, 'numeric_inconsistent')
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Test Case 6: Fixed but incorrect decimals
        result = self.data_smells.check_precision_consistency(df, 3, 'numeric_wrong_decimals')
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Test Case 7: Correct decimals
        result = self.data_smells.check_precision_consistency(df, 3, 'numeric_consistent')
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Test Case 8: Integer field
        result = self.data_smells.check_precision_consistency(df, 0, 'integer_field')
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Test Case 9: All numeric fields check
        result = self.data_smells.check_precision_consistency(df, 3, None)
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Test Case 10: Empty DataFrame
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_smells.check_precision_consistency(empty_df, 3, 'any_field')
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Test Case 11: All NaN values
        result = self.data_smells.check_precision_consistency(df, 3, 'all_nan')
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Test Case 12: Mixed integer and float values
        result = self.data_smells.check_precision_consistency(df, 1, 'mixed_types')
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        print_and_log("\nFinished testing check_precision_consistency function")
        print_and_log("")
