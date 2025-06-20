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
            self.execute_check_missing_invalid_value_consistency_SimpleTests
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

    def execute_check_missing_invalid_value_consistency_SimpleTests(self):
        """
        Execute simple tests for check_missing_invalid_value_consistency function.
        Tests various scenarios with simple test data.
        """
        print_and_log("")
        print_and_log("Testing check_missing_invalid_value_consistency function...")

        # Create test data with various cases - All arrays must have the same length (5)
        data = {
            'clean_column': ['value1', 'value2', 'value3', 'value4', 'value5'],
            'missing_values': ['value1', '', 'null', 'none', 'na'],
            'invalid_values': ['1.5', 'inf', '-inf', 'nan', '2.5'],
            'mixed_values': ['value1', 'inf', 'na', 'none', '-inf'],
            'custom_missing': ['MISSING', 'N/A', 'undefined', 'MISSING', 'N/A'],
            'custom_invalid': ['ERROR', 'INFINITY', 'NOT_NUMBER', 'ERROR', 'INFINITY'],
            'empty_column': ['', '', '', '', ''],
            'all_valid': ['1', '2', '3', '4', '5'],
            'case_sensitive': ['NA', 'Null', 'None', 'NA', 'Null'],
            'mixed_cases': ['INF', 'NaN', 'NULL', 'inf', 'nan']
        }
        df = pd.DataFrame(data)

        # Test cases
        print_and_log("\nStarting individual test cases...")

        # Test 1: Clean column with no missing/invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['MISSING'], ['', '?', '.', 'null', 'none', 'na'], 'clean_column')
        assert result is True, "Test Case 1 Failed"
        print_and_log("Test Case 1 Passed: Clean column check successful")

        # Test 2: Column with common missing values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['value1'], ['', '?', '.', 'null', 'none', 'na'], 'missing_values')
        assert result is False, "Test Case 2 Failed"
        print_and_log("Test Case 2 Passed: Missing values detected correctly")

        # Test 3: Column with invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['1.5'], ['inf', '-inf', 'nan'], 'invalid_values')
        assert result is False, "Test Case 3 Failed"
        print_and_log("Test Case 3 Passed: Invalid values detected correctly")

        # Test 4: Column with mixed values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['value1'], ['inf', '-inf', 'nan', 'na', 'none'], 'mixed_values')
        assert result is False, "Test Case 4 Failed"
        print_and_log("Test Case 4 Passed: Mixed values detected correctly")

        # Test 5: Custom missing values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['MISSING', 'N/A', 'undefined'], [''], 'custom_missing')
        assert result is True, "Test Case 5 Failed"
        print_and_log("Test Case 5 Passed: Custom missing values handled correctly")

        # Test 6: Custom invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, ['ERROR', 'INFINITY', 'NOT_NUMBER'], ['inf'], 'custom_invalid')
        assert result is True, "Test Case 6 Failed"
        print_and_log("Test Case 6 Passed: Custom invalid values handled correctly")

        # Test 7: Empty column
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [''], ['', '?', '.'], 'empty_column')
        assert result is True, "Test Case 7 Failed"
        print_and_log("Test Case 7 Passed: Empty column handled correctly")

        # Test 8: All valid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [], ['', 'null'], 'all_valid')
        assert result is True, "Test Case 8 Failed"
        print_and_log("Test Case 8 Passed: All valid values handled correctly")

        # Test 9: Case sensitive values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [], ['na', 'null', 'none'], 'case_sensitive')
        assert result is True, "Test Case 9 Failed"
        print_and_log("Test Case 9 Passed: Case sensitivity handled correctly")

        # Test 10: Mixed case values
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [], ['inf', 'nan', 'null'], 'mixed_cases')
        assert result is False, "Test Case 10 Failed"
        print_and_log("Test Case 10 Passed: Mixed case values handled correctly")

        # Test 11: Non-existent column
        with self.assertRaises(ValueError):
            self.data_smells.check_missing_invalid_value_consistency(
                df, [], [''], 'non_existent_column')
        print_and_log("Test Case 11 Passed: Non-existent column handled correctly")

        # Test 12: Empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.data_smells.check_missing_invalid_value_consistency(
                empty_df, [], [''], 'any_column')
        print_and_log("Test Case 12 Passed: Empty DataFrame handled correctly")

        # Test 13: None in lists
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [None], ['none'], 'missing_values')
        assert result is False, "Test Case 13 Failed"
        print_and_log("Test Case 13 Passed: None in lists handled correctly")

        # Test 14: Empty lists
        result = self.data_smells.check_missing_invalid_value_consistency(
            df, [], [], 'clean_column')
        assert result is True, "Test Case 14 Failed"
        print_and_log("Test Case 14 Passed: Empty lists handled correctly")

        # Test 15: Invalid input types
        with self.assertRaises(TypeError):
            self.data_smells.check_missing_invalid_value_consistency(
                df, "not_a_list", [''], 'clean_column')
        print_and_log("Test Case 15 Passed: Invalid input types handled correctly")

        # Test 16: Numeric values as strings
        data_numeric = {'numeric_column': ['1', '2', 'inf', '4', '5']}
        df_numeric = pd.DataFrame(data_numeric)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_numeric, [], ['inf'], 'numeric_column')
        assert result is False, "Test Case 16 Failed"
        print_and_log("Test Case 16 Passed: Numeric values as strings handled correctly")

        # Test 17: Special characters
        data_special = {'special_column': ['#N/A', '@@', '##', '@@', '#N/A']}
        df_special = pd.DataFrame(data_special)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_special, ['#N/A'], ['@@'], 'special_column')
        assert result is False, "Test Case 17 Failed"
        print_and_log("Test Case 17 Passed: Special characters handled correctly")

        # Test 18: Whitespace values
        data_whitespace = {'whitespace_column': [' ', '  ', '\t', '\n', ' ']}
        df_whitespace = pd.DataFrame(data_whitespace)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_whitespace, [' ', '  '], ['\t', '\n'], 'whitespace_column')
        assert result is False, "Test Case 18 Failed"

        # Test 19: Unicode characters
        data_unicode = {'unicode_column': ['á', 'é', 'í', 'ó', 'ú']}
        df_unicode = pd.DataFrame(data_unicode)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_unicode, ['á'], ['ñ'], 'unicode_column')
        assert result is True, "Test Case 19 Failed"
        print_and_log("Test Case 19 Passed: Unicode characters handled correctly")

        # Test 20: Large number of unique values
        large_data = {'large_column': ['val' + str(i) for i in range(4)] + ['inf']}
        df_large = pd.DataFrame(large_data)
        result = self.data_smells.check_missing_invalid_value_consistency(
            df_large, [], ['inf'], 'large_column')
        assert result is False, "Test Case 20 Failed"
        print_and_log("Test Case 20 Passed: Large number of unique values handled correctly")

        print_and_log("\nFinished testing check_missing_invalid_value_consistency function")
        print_and_log("-----------------------------------------------------------")
