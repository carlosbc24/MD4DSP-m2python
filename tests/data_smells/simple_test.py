# Importing libraries
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm
import functions.data_smells as data_smells
from helpers.enumerations import DataType

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
            self.execute_check_missing_invalid_value_consistency_SimpleTests,
            self.execute_check_integer_as_floating_point_SimpleTests,
            self.execute_check_types_as_string_SimpleTests,
            self.execute_check_special_character_spacing_SimpleTests,
            self.execute_check_suspect_distribution_SimpleTests,
            self.execute_check_suspect_precision_SimpleTests,
            self.execute_check_date_as_datetime_SimpleTests,
            self.execute_check_separating_consistency_SimpleTests,
            self.execute_check_date_time_consistency_SimpleTests,
            self.execute_check_ambiguous_datetime_format_SimpleTests,
            self.execute_check_suspect_date_value_SimpleTests
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

    def execute_check_integer_as_floating_point_SimpleTests(self):
        """
        Execute simple tests for check_integer_as_floating_point function.
        Tests the following cases:
        1. Non-existent field
        2. Non-numeric field
        3. Integer field represented as float (smell)
        4. Float field (no smell)
        5. Integer field (no smell)
        6. Empty DataFrame
        7. Column with all NaN values
        8. Mixed integer and float values (smell)
        """
        print_and_log("")
        print_and_log("Testing check_integer_as_floating_point function...")

        # Create test data
        data = {
            'int_as_float': [1.0, 2.0, 3.0],
            'float_field': [1.1, 2.5, 3.8],
            'integer_field': [1, 2, 3],
            'non_numeric': ['a', 'b', 'c'],
            'all_nan': [np.nan, np.nan, np.nan],
            'mixed_int_float': [1.0, 2.5, 3.0]
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test Case 1: Non-existent field
        expected_exception = ValueError
        with self.assertRaises(expected_exception):
            self.data_smells.check_integer_as_floating_point(df, 'non_existent_field')
        print_and_log("Test Case 1 Passed: Expected ValueError, got ValueError")

        # Test Case 2: Non-numeric field
        # Assuming the function handles non-numeric gracefully
        result = self.data_smells.check_integer_as_floating_point(df, 'non_numeric')
        self.assertTrue(result)
        print_and_log("Test Case 2 Passed: Expected no smell for non-numeric, got no smell")

        # Test Case 3: Integer field represented as float (smell)
        result = self.data_smells.check_integer_as_floating_point(df, 'int_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 3 Passed: Expected smell detection, got smell detection")

        # Test Case 4: Float field (no smell)
        result = self.data_smells.check_integer_as_floating_point(df, 'float_field')
        self.assertTrue(result)
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test Case 5: Integer field (no smell)
        result = self.data_smells.check_integer_as_floating_point(df, 'integer_field')
        self.assertTrue(result)
        print_and_log("Test Case 5 Passed: Expected no smell, got no smell")

        # Test Case 6: Empty DataFrame with specific column (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_integer_as_floating_point(empty_df, 'any_column')
        print_and_log("Test Case 6 Passed: Expected ValueError for empty DataFrame with specific column, got ValueError")

        # Test Case 7: Column with all NaN values
        result = self.data_smells.check_integer_as_floating_point(df, 'all_nan')
        self.assertTrue(result)
        print_and_log("Test Case 7 Passed: Expected no smell for all NaN column, got no smell")

        # Test Case 8: Mixed integer and float values (no smell)
        result = self.data_smells.check_integer_as_floating_point(df, 'mixed_int_float')
        self.assertTrue(result)
        print_and_log("Test Case 8 Passed: Expected no smell detection for mixed types, got no smell detection")

        # Test Case 9: Float column with very small decimals (considered float, no smell)
        data_9 = {'float_with_small_decimals': [1.0001, 2.0002, 3.0003]}
        df_9 = pd.DataFrame(data_9)
        result = self.data_smells.check_integer_as_floating_point(df_9, 'float_with_small_decimals')
        self.assertTrue(result)
        print_and_log("Test Case 9 Passed: Expected no smell for float with small decimals, got no smell")

        # Test Case 10: Single value integer as float (smell)
        data_10 = {'single_int_as_float': [42.0]}
        df_10 = pd.DataFrame(data_10)
        result = self.data_smells.check_integer_as_floating_point(df_10, 'single_int_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 10 Passed: Expected smell for single integer as float, got smell")

        # Test Case 11: Large integer values as float (smell)
        data_11 = {'large_int_as_float': [1000000.0, 2000000.0, 3000000.0]}
        df_11 = pd.DataFrame(data_11)
        result = self.data_smells.check_integer_as_floating_point(df_11, 'large_int_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 11 Passed: Expected smell for large integers as float, got smell")

        # Test Case 12: Negative integers as float (smell)
        data_12 = {'negative_int_as_float': [-1.0, -2.0, -3.0]}
        df_12 = pd.DataFrame(data_12)
        result = self.data_smells.check_integer_as_floating_point(df_12, 'negative_int_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 12 Passed: Expected smell for negative integers as float, got smell")

        # Test Case 13: Zero values as float (smell)
        data_13 = {'zeros_as_float': [0.0, 0.0, 0.0]}
        df_13 = pd.DataFrame(data_13)
        result = self.data_smells.check_integer_as_floating_point(df_13, 'zeros_as_float')
        self.assertFalse(result)
        print_and_log("Test Case 13 Passed: Expected smell for zeros as float, got smell")

        # Test Case 14: String column (no smell)
        data_14 = {'string_column': ['hello', 'world', 'test']}
        df_14 = pd.DataFrame(data_14)
        result = self.data_smells.check_integer_as_floating_point(df_14, 'string_column')
        self.assertTrue(result)
        print_and_log("Test Case 14 Passed: Expected no smell for string column, got no smell")

        # Test Case 15: Check all columns at once (smell present)
        data_15 = {
            'good_float': [1.1, 2.2, 3.3],
            'bad_float': [1.0, 2.0, 3.0],
            'string_col': ['a', 'b', 'c']
        }
        df_15 = pd.DataFrame(data_15)
        result = self.data_smells.check_integer_as_floating_point(df_15)  # Check all columns
        self.assertFalse(result)
        print_and_log("Test Case 15 Passed: Expected smell when checking all columns, got smell")

    def execute_check_types_as_string_SimpleTests(self):
        """
        Execute simple tests for check_types_as_string function.
        Tests the following cases:
        1. All values are integer strings (should warn)
        2. All values are float strings (should warn)
        3. All values are date strings (should warn)
        4. All values are time strings (should warn)
        5. All values are datetime strings (should warn)
        6. Mixed string values (should not warn)
        7. Type mismatch (should raise TypeError)
        8. Non-existent field (should raise ValueError)
        9. Unknown expected_type (should raise ValueError)
        """
        print_and_log("")
        print_and_log("Testing check_types_as_string function...")

        # Test data
        df = pd.DataFrame({
            'int_str': ['1', '2', '-3'],
            'float_str': ['1.1', '-2.2', '+3.3'],
            'date_str': ['2024-06-24', '2023-01-01', '2022-12-31'],
            'time_str': ['12:34', '23:59:59', '11:11 AM'],
            'datetime_str': ['2024-06-24 12:34:56', '2023-01-01 23:59', 'March 8, 2024 11:59 PM'],
            'mixed_str': ['abc', '123', '2024-06-24'],
            'true_int': [1, 2, 3]
        })

        # 1. All integer strings
        result = self.data_smells.check_types_as_string(df, 'int_str', DataType.STRING)
        assert result is False, "Test Case 1 Failed: Should warn for integer as string"
        print_and_log("Test Case 1 Passed: Integer as string detected")

        # 2. All float strings
        result = self.data_smells.check_types_as_string(df, 'float_str', DataType.STRING)
        assert result is False, "Test Case 2 Failed: Should warn for float as string"
        print_and_log("Test Case 2 Passed: Float as string detected")

        # 3. All date strings
        result = self.data_smells.check_types_as_string(df, 'date_str', DataType.STRING)
        assert result is False, "Test Case 3 Failed: Should warn for date as string"
        print_and_log("Test Case 3 Passed: Date as string detected")

        # 4. All time strings
        result = self.data_smells.check_types_as_string(df, 'time_str', DataType.STRING)
        assert result is False, "Test Case 4 Failed: Should warn for time as string"
        print_and_log("Test Case 4 Passed: Time as string detected")

        # 5. All datetime strings
        result = self.data_smells.check_types_as_string(df, 'datetime_str', DataType.STRING)
        assert result is False, "Test Case 5 Failed: Should warn for datetime as string"
        print_and_log("Test Case 5 Passed: Datetime as string detected")

        # 6. Mixed string values (should not warn)
        result = self.data_smells.check_types_as_string(df, 'mixed_str', DataType.STRING)
        assert result is True, "Test Case 6 Failed: Should not warn for mixed string values"
        print_and_log("Test Case 6 Passed: Mixed string values handled correctly")

        # 7. All integer strings as float (should not warn)
        self.data_smells.check_types_as_string(df, 'int_str', DataType.INTEGER)
        assert result is True, "Test Case 7 Failed: Should not warn for integer strings as float"
        print_and_log("Test Case 7 Passed: Integer strings as float handled correctly")

        # 8. Non-existent field (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_types_as_string(df, 'no_field', DataType.STRING)
        print_and_log("Test Case 8 Passed: ValueError raised for non-existent field")

        # 9. Unknown expected_type (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_types_as_string(df, 'int_str', "CustomType")
        print_and_log("Test Case 9 Passed: ValueError raised for unknown expected_type")

        # 10. Column true_int with expected_type String (should warn)
        result = self.data_smells.check_types_as_string(df, 'true_int', DataType.STRING)
        assert result is False, "Test Case 10 Failed: Should warn for integer column as string"
        print_and_log("Test Case 10 Passed: Integer column as string detected")

        # 11. Empty column (should not warn, just return True)
        df['empty'] = [''] * len(df)
        result = self.data_smells.check_types_as_string(df, 'empty', DataType.STRING)
        assert result is True, "Test Case 11 Failed: Should not warn for empty column"
        print_and_log("Test Case 11 Passed: Empty column handled correctly")

        # 12. Column with boolean strings (should not warn)
        df['bool_str'] = ['True', 'False', 'True']
        result = self.data_smells.check_types_as_string(df, 'bool_str', DataType.STRING)
        assert result is True, "Test Case 12 Failed: Should not warn for boolean strings"
        print_and_log("Test Case 12 Passed: Boolean strings handled correctly")

        # 13. Column with only spaces (should not warn)
        df['spaces'] = ['   ', ' ', '  ']
        result = self.data_smells.check_types_as_string(df, 'spaces', DataType.STRING)
        assert result is True, "Test Case 13 Failed: Should not warn for spaces only"
        print_and_log("Test Case 13 Passed: Spaces only handled correctly")

        # 14. Column with special characters (should not warn)
        df['special'] = ['@', '#', '$']
        result = self.data_smells.check_types_as_string(df, 'special', DataType.STRING)
        assert result is True, "Test Case 14 Failed: Should not warn for special characters"
        print_and_log("Test Case 14 Passed: Special characters handled correctly")

        # 15. Column with single value (should not warn)
        df['single'] = ['unique'] * len(df)
        result = self.data_smells.check_types_as_string(df, 'single', DataType.STRING)
        assert result is True, "Test Case 15 Failed: Should not warn for single value column"
        print_and_log("Test Case 15 Passed: Single value column handled correctly")

        # 16. Column with repeated integer strings (should warn)
        df['repeated_int'] = ['7'] * len(df)
        result = self.data_smells.check_types_as_string(df, 'repeated_int', DataType.STRING)
        assert result is False, "Test Case 16 Failed: Should warn for repeated integer strings"
        print_and_log("Test Case 16 Passed: Repeated integer strings detected")

        # 17. Column with repeated float strings (should warn)
        df['repeated_float'] = ['3.14'] * len(df)
        result = self.data_smells.check_types_as_string(df, 'repeated_float', DataType.STRING)
        assert result is False, "Test Case 17 Failed: Should warn for repeated float strings"
        print_and_log("Test Case 17 Passed: Repeated float strings detected")

        # 18. Column con fechas en formato alternativo (debería advertir)
        df['alt_date'] = ['08/03/2024', '07/07/2023', '25/12/2022']
        result = self.data_smells.check_types_as_string(df, 'alt_date', DataType.STRING)
        assert result is False, "Test Case 18 Failed: Should warn for alternative date format as string"
        print_and_log("Test Case 18 Passed: Alternative date format as string detected")

        # 19. Column con horas en formato alternativo (debería advertir)
        df['alt_time'] = ['11:59 PM', '10:00 AM', '01:23 PM']
        result = self.data_smells.check_types_as_string(df, 'alt_time', DataType.STRING)
        assert result is False, "Test Case 19 Failed: Should warn for alternative time format as string"
        print_and_log("Test Case 19 Passed: Alternative time format as string detected")

        print_and_log("\nFinished testing check_types_as_string function")
        print_and_log("")

    def execute_check_special_character_spacing_SimpleTests(self):
        """
        Execute simple tests for check_special_character_spacing function.
        Tests the following cases:
        1. Non-existent field
        2. String field with clean text (no smell)
        3. String field with uppercase letters (smell)
        4. String field with accents (smell)
        5. String field with special characters (smell)
        6. String field with extra spaces (smell)
        7. String field with mixed issues (smell)
        8. Numeric field (no smell)
        9. Empty DataFrame
        10. Column with all NaN values
        11. Column with empty strings (no smell)
        12. Column with single character issues (smell)
        13. Column with numbers as strings (no smell)
        14. Column with mixed clean and dirty text (smell)
        15. Check all columns at once (smell present)
        """
        print_and_log("")
        print_and_log("Testing check_special_character_spacing function...")

        # Create test data
        data = {
            'clean_text': ['hello world', 'test case', 'simple text'],
            'uppercase_text': ['Hello World', 'TEST CASE', 'Simple Text'],
            'accented_text': ['café', 'niño', 'résumé'],
            'special_chars': ['hello@world', 'test#case', 'simple!text'],
            'extra_spaces': ['hello  world', 'test   case', 'simple    text'],
            'mixed_issues': ['Café@Home  ', 'TEST#Case   ', 'Résumé!Final  '],
            'numeric_field': [1, 2, 3],
            'all_nan': [np.nan, np.nan, np.nan],
            'empty_column': ['', '', ''],
            'single_char_issues': ['A', '@', ' '],
            'numbers_as_strings': ['123', '456', '789'],
            'mixed_clean_dirty': ['clean text', 'Dirty@Text  ', 'normal']
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test Case 1: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_special_character_spacing(df, 'non_existent_field')
        print_and_log("Test Case 1 Passed: Expected ValueError, got ValueError")

        # Test Case 2: String field with clean text (no smell)
        result = self.data_smells.check_special_character_spacing(df, 'clean_text')
        self.assertTrue(result)
        print_and_log("Test Case 2 Passed: Expected no smell for clean text, got no smell")

        # Test Case 3: String field with uppercase letters (smell)
        result = self.data_smells.check_special_character_spacing(df, 'uppercase_text')
        self.assertFalse(result)
        print_and_log("Test Case 3 Passed: Expected smell for uppercase text, got smell")

        # Test Case 4: String field with accents (smell)
        result = self.data_smells.check_special_character_spacing(df, 'accented_text')
        self.assertFalse(result)
        print_and_log("Test Case 4 Passed: Expected smell for accented text, got smell")

        # Test Case 5: String field with special characters (smell)
        result = self.data_smells.check_special_character_spacing(df, 'special_chars')
        self.assertFalse(result)
        print_and_log("Test Case 5 Passed: Expected smell for special characters, got smell")

        # Test Case 6: String field with extra spaces (smell)
        result = self.data_smells.check_special_character_spacing(df, 'extra_spaces')
        self.assertFalse(result)
        print_and_log("Test Case 6 Passed: Expected smell for extra spaces, got smell")

        # Test Case 7: String field with mixed issues (smell)
        result = self.data_smells.check_special_character_spacing(df, 'mixed_issues')
        self.assertFalse(result)
        print_and_log("Test Case 7 Passed: Expected smell for mixed issues, got smell")

        # Test Case 8: Numeric field (no smell)
        result = self.data_smells.check_special_character_spacing(df, 'numeric_field')
        self.assertTrue(result)
        print_and_log("Test Case 8 Passed: Expected no smell for numeric field, got no smell")

        # Test Case 9: Empty DataFrame with specific column (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_special_character_spacing(empty_df, 'any_column')
        print_and_log("Test Case 9 Passed: Expected ValueError for empty DataFrame with specific column, got ValueError")

        # Test Case 10: Column with all NaN values
        result = self.data_smells.check_special_character_spacing(df, 'all_nan')
        self.assertTrue(result)
        print_and_log("Test Case 10 Passed: Expected no smell for all NaN column, got no smell")

        # Test Case 11: Column with empty strings (no smell)
        result = self.data_smells.check_special_character_spacing(df, 'empty_column')
        self.assertTrue(result)
        print_and_log("Test Case 11 Passed: Expected no smell for empty strings, got no smell")

        # Test Case 12: Column with single character issues (smell)
        result = self.data_smells.check_special_character_spacing(df, 'single_char_issues')
        self.assertFalse(result)
        print_and_log("Test Case 12 Passed: Expected smell for single character issues, got smell")

        # Test Case 13: Column with numbers as strings (no smell)
        result = self.data_smells.check_special_character_spacing(df, 'numbers_as_strings')
        self.assertTrue(result)
        print_and_log("Test Case 13 Passed: Expected no smell for numbers as strings, got no smell")

        # Test Case 14: Column with mixed clean and dirty text (smell)
        result = self.data_smells.check_special_character_spacing(df, 'mixed_clean_dirty')
        self.assertFalse(result)
        print_and_log("Test Case 14 Passed: Expected smell for mixed clean/dirty text, got smell")

        # Test Case 15: Check all columns at once (smell present)
        result = self.data_smells.check_special_character_spacing(df)  # Check all columns
        self.assertFalse(result)
        print_and_log("Test Case 15 Passed: Expected smell when checking all columns, got smell")

    def execute_check_suspect_distribution_SimpleTests(self):
        """
        Execute simple tests for check_suspect_distribution function.
        Tests the following cases:
        1. Invalid min/max parameters (non-numeric)
        2. Invalid range (min > max)
        3. Non-existent field
        4. Values within range (no smell)
        5. Values outside range - too high (smell)
        6. Values outside range - too low (smell)
        7. Values outside range - both ends (smell)
        8. Non-numeric field (no smell)
        9. Empty DataFrame
        10. Column with all NaN values
        11. Mixed values in and out of range (smell)
        12. Exact boundary values (no smell)
        13. Float precision at boundaries (no smell)
        14. Large dataset with outliers (smell)
        15. Check all columns at once (smell present)
        """
        print_and_log("")
        print_and_log("Testing check_suspect_distribution function...")

        # Test Case 1: Invalid min/max parameters (non-numeric)
        df_dummy = pd.DataFrame({'test': [1, 2, 3]})
        with self.assertRaises(TypeError):
            self.data_smells.check_suspect_distribution(df_dummy, "invalid", 10.0, 'test')
        print_and_log("Test Case 1 Passed: Expected TypeError for non-numeric parameters, got TypeError")

        # Test Case 2: Invalid range (min > max)
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_distribution(df_dummy, 10.0, 5.0, 'test')
        print_and_log("Test Case 2 Passed: Expected ValueError for invalid range, got ValueError")

        # Create test data
        data = {
            'values_in_range': [1.0, 2.5, 4.0, 3.2, 2.8],
            'values_too_high': [1.0, 2.0, 6.0, 3.0, 2.5],
            'values_too_low': [-1.0, 2.0, 3.0, 4.0, 2.5],
            'values_both_ends': [-1.0, 2.0, 6.0, 3.0, 2.5],
            'non_numeric': ['a', 'b', 'c', 'd', 'e'],
            'all_nan': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'mixed_in_out': [1.0, 2.0, 3.0, 6.0, 2.5],
            'boundary_values': [0.0, 2.5, 5.0, 1.0, 4.0],
            'float_precision': [0.000001, 2.5, 4.999999, 3.0, 2.0]
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Define range for tests: 0.0 to 5.0
        min_val, max_val = 0.0, 5.0

        # Test Case 3: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_distribution(df, min_val, max_val, 'non_existent_field')
        print_and_log("Test Case 3 Passed: Expected ValueError for non-existent field, got ValueError")

        # Test Case 4: Values within range (no smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'values_in_range')
        self.assertTrue(result)
        print_and_log("Test Case 4 Passed: Expected no smell for values in range, got no smell")

        # Test Case 5: Values outside range - too high (smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'values_too_high')
        self.assertFalse(result)
        print_and_log("Test Case 5 Passed: Expected smell for values too high, got smell")

        # Test Case 6: Values outside range - too low (smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'values_too_low')
        self.assertFalse(result)
        print_and_log("Test Case 6 Passed: Expected smell for values too low, got smell")

        # Test Case 7: Values outside range - both ends (smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'values_both_ends')
        self.assertFalse(result)
        print_and_log("Test Case 7 Passed: Expected smell for values at both ends, got smell")

        # Test Case 8: Non-numeric field (no smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'non_numeric')
        self.assertTrue(result)
        print_and_log("Test Case 8 Passed: Expected no smell for non-numeric field, got no smell")

        # Test Case 9: Empty DataFrame with specific column (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_distribution(empty_df, min_val, max_val, 'any_column')
        print_and_log("Test Case 9 Passed: Expected ValueError for empty DataFrame with specific column, got ValueError")

        # Test Case 10: Column with all NaN values
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'all_nan')
        self.assertTrue(result)
        print_and_log("Test Case 10 Passed: Expected no smell for all NaN column, got no smell")

        # Test Case 11: Mixed values in and out of range (smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'mixed_in_out')
        self.assertFalse(result)
        print_and_log("Test Case 11 Passed: Expected smell for mixed in/out values, got smell")

        # Test Case 12: Exact boundary values (no smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'boundary_values')
        self.assertTrue(result)
        print_and_log("Test Case 12 Passed: Expected no smell for boundary values, got no smell")

        # Test Case 13: Float precision at boundaries (no smell)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val, 'float_precision')
        self.assertTrue(result)
        print_and_log("Test Case 13 Passed: Expected no smell for float precision at boundaries, got no smell")

        # Test Case 14: Large dataset with outliers (smell)
        large_data = pd.DataFrame({
            'large_dataset': [2.0] * 100 + [10.0]  # 100 normal values + 1 outlier
        })
        result = self.data_smells.check_suspect_distribution(large_data, min_val, max_val, 'large_dataset')
        self.assertFalse(result)
        print_and_log("Test Case 14 Passed: Expected smell for large dataset with outliers, got smell")

        # Test Case 15: Check all columns at once (smell present)
        result = self.data_smells.check_suspect_distribution(df, min_val, max_val)  # Check all columns
        self.assertFalse(result)
        print_and_log("Test Case 15 Passed: Expected smell when checking all columns, got smell")

    def execute_check_suspect_precision_SimpleTests(self):
        """
        Execute simple tests for check_suspect_precision function.
        Tests cases for decimal and numeric precision issues
        """
        print_and_log("")
        print_and_log("Testing check_suspect_precision function...")

        # Create test data
        data = {
            'non_significant': [1.0000, 2.0000, 3.0000],
            'proper_precision': [1.23, 2.34, 3.45],
            'mixed_precision': [1.230, 2.0, 3.400],
            'non_float': ['a', 'b', 'c'],
            'with_nan': [1.200, np.nan, 3.400],
            'with_none': [1.200, None, 3.400],
            'large_numbers': [1000000.000, 2000000.000, 3000000.000],
            'small_numbers': [0.001000, 0.002000, 0.003000],
            'negative_numbers': [-1.200, -2.300, -3.400],
            'scientific': [1.23e-4, 2.34e-4, 3.45e-4],
            'mixed_significant': [1.23, 2.000, 3.45000]
        }
        df = pd.DataFrame(data)
        empty_df = pd.DataFrame()

        # Test 1: Column with non-significant zeros
        result = self.data_smells.check_suspect_precision(df, 'non_significant')
        assert result is True, "Test Case 1 Failed: Should'nt detect smell in non-significant zeros"
        print_and_log("Test Case 1 Passed: Non-significant zeros have not been detected")

        # Test 2: Column with proper precision
        result = self.data_smells.check_suspect_precision(df, 'proper_precision')
        assert result is True, "Test Case 2 Failed: Should not detect smell in proper precision"
        print_and_log("Test Case 2 Passed: Proper precision accepted")

        # Test 3: Column with mixed precision
        result = self.data_smells.check_suspect_precision(df, 'mixed_precision')
        assert result is True, "Test Case 3 Failed: Should'nt detect mixed precision as right zeros aren't significant"
        print_and_log("Test Case 3 Passed: Mixed precision have not been detected")

        # Test 4: Non-float column (no smell)
        result = self.data_smells.check_suspect_precision(df, 'non_float')
        assert result is True, "Test Case 4 Failed: Should ignore non-float column"
        print_and_log("Test Case 4 Passed: Non-float column ignored")

        # Test 6: Column with NaN values
        result = self.data_smells.check_suspect_precision(df, 'with_nan')
        assert result is True, "Test Case 6 Failed: Should not detect smell despite NaN values"
        print_and_log("Test Case 6 Passed: NaN values handled correctly")

        # Test 7: Column with None values
        result = self.data_smells.check_suspect_precision(df, 'with_none')
        assert result is True, "Test Case 7 Failed: Should not detect smell despite None values"
        print_and_log("Test Case 7 Passed: None values handled correctly")

        # Test 8: Column with large numbers
        result = self.data_smells.check_suspect_precision(df, 'large_numbers')
        assert result is True, "Test Case 8 Failed: Should not detect non-significant digits in large numbers"
        print_and_log("Test Case 8 Passed: Large numbers handled correctly")

        # Test 9: Column with small numbers
        result = self.data_smells.check_suspect_precision(df, 'small_numbers')
        assert result is True, "Test Case 9 Failed: Should not detect non-significant digits in small numbers"
        print_and_log("Test Case 9 Passed: Small numbers handled correctly")

        # Test 10: Column with negative numbers
        result = self.data_smells.check_suspect_precision(df, 'negative_numbers')
        assert result is True, "Test Case 10 Failed: Should not detect non-significant digits in negative numbers"
        print_and_log("Test Case 10 Passed: Negative numbers handled correctly")

        # Test 11: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_suspect_precision(df, 'non_existent')
        print_and_log("Test Case 11 Passed: Non-existent field handled correctly")

        # Test 12: Empty DataFrame
        result = self.data_smells.check_suspect_precision(empty_df)
        assert result is True, "Test Case 12 Failed: Should handle empty DataFrame"
        print_and_log("Test Case 12 Passed: Empty DataFrame handled correctly")

        # Test 13: Column with scientific notation
        result = self.data_smells.check_suspect_precision(df, 'scientific')
        assert result is True, "Test Case 13 Failed: Should handle scientific notation correctly"
        print_and_log("Test Case 13 Passed: Scientific notation handled correctly")

        # Test 14: Column with mixed significant and non-significant digits
        result = self.data_smells.check_suspect_precision(df, 'mixed_significant')
        assert result is True, "Test Case 14 Failed: Should not detect mixed significant digits"
        print_and_log("Test Case 14 Passed: Mixed significant digits detected")

        # Test 15: Check all float columns at once
        result = self.data_smells.check_suspect_precision(df)
        assert result is True, "Test Case 15 Failed: Should not detect smell in at least one column"
        print_and_log("Test Case 15 Passed: All columns check successful")

        # New tests for suspect precision
        # Test 16: Periodic decimal numbers
        df_periodic = pd.DataFrame({
            'periodic': [1/3, 2/3, 1/6]  # Will generate periodic decimals
        })
        result = self.data_smells.check_suspect_precision(df_periodic, 'periodic')
        assert result is False, "Test Case 16 Failed: Should detect smell for periodic decimals"
        print_and_log("Test Case 16 Passed: Periodic decimals detected")

        # Test 17: Long float chain (>15 decimals)
        df_long = pd.DataFrame({
            'long_decimals': [
                1.123456789012345,
                2.123456789012345,
                3.123456789012345
            ]
        })
        result = self.data_smells.check_suspect_precision(df_long, 'long_decimals')
        assert result is False, "Test Case 17 Failed: Should detect smell for long decimals"
        print_and_log("Test Case 17 Passed: Long decimal chain detected")

        # Test 18: Mixed periodic and non-periodic
        df_mixed_periodic = pd.DataFrame({
            'mixed_periodic': [1/3, 0.5, 1/6]
        })
        result = self.data_smells.check_suspect_precision(df_mixed_periodic, 'mixed_periodic')
        assert result is False, "Test Case 18 Failed: Should detect smell for mixed periodic decimals"
        print_and_log("Test Case 18 Passed: Mixed periodic decimals detected")

        # Test 19: Pure periodic decimals
        df_pure_periodic = pd.DataFrame({
            'pure_periodic': [1/11, 2/11, 3/11]  # Pure periodic decimal
        })
        result = self.data_smells.check_suspect_precision(df_pure_periodic, 'pure_periodic')
        assert result is False, "Test Case 19 Failed: Should detect smell for pure periodic decimals"
        print_and_log("Test Case 19 Passed: Pure periodic decimals detected")

        # Test 20: Mixed periodic decimals
        df_mixed_type_periodic = pd.DataFrame({
            'mixed_type_periodic': [1/6, 1/7, 1/13]  # Different types of periodic decimals
        })
        result = self.data_smells.check_suspect_precision(df_mixed_type_periodic, 'mixed_type_periodic')
        assert result is False, "Test Case 20 Failed: Should detect smell for mixed type periodic decimals"
        print_and_log("Test Case 20 Passed: Mixed type periodic decimals detected")

        # Test 21: Very small numbers close to floating point precision limit
        df_tiny = pd.DataFrame({
            'tiny_numbers': [1e-15, 2e-15, 3e-15]
        })
        result = self.data_smells.check_suspect_precision(df_tiny, 'tiny_numbers')
        assert result is True, "Test Case 21 Failed: Should not detect smell for very small numbers"
        print_and_log("Test Case 21 Passed: Very small numbers handled correctly")

        # Test 22: Numbers requiring high precision arithmetic
        df_high_precision = pd.DataFrame({
            'high_precision': [np.pi, np.e, np.sqrt(2)]
        })
        result = self.data_smells.check_suspect_precision(df_high_precision, 'high_precision')
        assert result is False, "Test Case 22 Failed: Should detect smell for high precision numbers"
        print_and_log("Test Case 22 Passed: High precision numbers detected")

        # Test 23: Numbers with potential rounding errors
        df_rounding = pd.DataFrame({
            'rounding_errors': [0.1 + 0.2, 0.7 + 0.1, 0.3 + 0.6]  # Known floating point precision issues
        })
        result = self.data_smells.check_suspect_precision(df_rounding, 'rounding_errors')
        assert result is False, "Test Case 23 Failed: Should detect smell for numbers with rounding errors"
        print_and_log("Test Case 23 Passed: Rounding errors detected")

        # Test 24: Recurring decimal patterns
        df_recurring = pd.DataFrame({
            'recurring': [1/7, 2/7, 3/7]  # Numbers with recurring decimal patterns
        })
        result = self.data_smells.check_suspect_precision(df_recurring, 'recurring')
        assert result is False, "Test Case 24 Failed: Should detect smell for recurring decimals"
        print_and_log("Test Case 24 Passed: Recurring decimals detected")

        # Test 25: Numbers requiring arbitrary precision
        df_arbitrary = pd.DataFrame({
            'arbitrary_precision': [np.sqrt(3), np.sqrt(5), np.sqrt(7)]
        })
        result = self.data_smells.check_suspect_precision(df_arbitrary, 'arbitrary_precision')
        assert result is False, "Test Case 25 Failed: Should detect smell for arbitrary precision numbers"
        print_and_log("Test Case 25 Passed: Arbitrary precision numbers detected")

        # Test 26: Numbers with floating point representation issues
        df_float_issues = pd.DataFrame({
            'float_issues': [0.1234567890123456, 1.2345678901234567, 2.3456789012345678]
        })
        result = self.data_smells.check_suspect_precision(df_float_issues, 'float_issues')
        assert result is False, "Test Case 26 Failed: Should detect smell for float representation issues"
        print_and_log("Test Case 26 Passed: Float representation issues detected")

        # Test 27: Numbers requiring extended precision
        df_extended = pd.DataFrame({
            'extended_precision': [np.exp(1), np.log(10), np.sin(np.pi/6)]
        })
        result = self.data_smells.check_suspect_precision(df_extended, 'extended_precision')
        assert result is False, "Test Case 27 Failed: Should detect smell for extended precision numbers"
        print_and_log("Test Case 27 Passed: Extended precision numbers detected")

        # Test 28: Irrational numbers
        df_irrational = pd.DataFrame({
            'irrational': [np.pi, np.e, np.sqrt(2)]
        })
        result = self.data_smells.check_suspect_precision(df_irrational, 'irrational')
        assert result is False, "Test Case 28 Failed: Should detect smell for irrational numbers"
        print_and_log("Test Case 28 Passed: Irrational numbers detected")

        # Test 29: Numbers with potential cancellation errors
        df_cancellation = pd.DataFrame({
            'cancellation': [(1e20 + 1) - 1e20, (1e15 + 1) - 1e15, (1e10 + 1) - 1e10]
        })
        result = self.data_smells.check_suspect_precision(df_cancellation, 'cancellation')
        assert result is True, "Test Case 29 Failed: Should not detect smell for cancellation errors"
        print_and_log("Test Case 29 Passed: Cancellation errors handled correctly")

        # Test 30: Numbers with precision accumulation issues
        df_accumulation = pd.DataFrame({
            'accumulation': [sum([0.1] * 10), sum([0.1] * 100), sum([0.1] * 1000)]
        })
        result = self.data_smells.check_suspect_precision(df_accumulation, 'accumulation')
        assert result is False, "Test Case 30 Failed: Should detect smell for precision accumulation"
        print_and_log("Test Case 30 Passed: Precision accumulation issues detected")

        print_and_log("\nFinished testing check_suspect_precision function")
        print_and_log("")

    def execute_check_date_as_datetime_SimpleTests(self):
        """
        Execute simple tests for check_date_as_datetime function.
        Tests various scenarios with different datetime data.
        """
        print_and_log("")
        print_and_log("Testing check_date_as_datetime function...")

        # Test 1: Create a DataFrame with pure date values (should detect smell)
        df_dates = pd.DataFrame({
            'pure_dates': pd.date_range('2024-01-01', periods=5, freq='D')
        })
        result = self.data_smells.check_date_as_datetime(df_dates, 'pure_dates')
        assert result is False, "Test Case 1 Failed: Should detect smell for pure dates"
        print_and_log("Test Case 1 Passed: Date smell detected correctly")

        # Test 2: Create a DataFrame with mixed times (no smell)
        df_mixed = pd.DataFrame({
            'mixed_times': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-02 15:45:30'),
                pd.Timestamp('2024-01-03 08:20:15')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_mixed, 'mixed_times')
        assert result is True, "Test Case 2 Failed: Should not detect smell for mixed times"
        print_and_log("Test Case 2 Passed: No smell detected for mixed times")

        # Test 3: Create a DataFrame with midnight times (should detect smell)
        df_midnight = pd.DataFrame({
            'midnight_times': pd.date_range('2024-01-01', periods=3, freq='D')
        })
        result = self.data_smells.check_date_as_datetime(df_midnight, 'midnight_times')
        assert result is False, "Test Case 3 Failed: Should detect smell for midnight times"
        print_and_log("Test Case 3 Passed: Smell detected for midnight times")

        # Test 4: Test with non-datetime column
        df_non_datetime = pd.DataFrame({
            'strings': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        result = self.data_smells.check_date_as_datetime(df_non_datetime, 'strings')
        assert result is True, "Test Case 4 Failed: Should not detect smell for non-datetime column"
        print_and_log("Test Case 4 Passed: No smell detected for non-datetime column")

        # Test 5: Test with empty DataFrame
        df_empty = pd.DataFrame()
        result = self.data_smells.check_date_as_datetime(df_empty)
        assert result is True, "Test Case 5 Failed: Should not detect smell for empty DataFrame"
        print_and_log("Test Case 5 Passed: No smell detected for empty DataFrame")

        # Test 6: Test with column containing NaN values
        df_with_nan = pd.DataFrame({
            'datetime_with_nan': [pd.Timestamp('2024-01-01'), np.nan, pd.Timestamp('2024-01-03')]
        })
        result = self.data_smells.check_date_as_datetime(df_with_nan, 'datetime_with_nan')
        assert result is False, "Test Case 6 Failed: Should detect smell for dates with NaN"
        print_and_log("Test Case 6 Passed: Smell detected correctly with NaN values")

        # Test 7: Test with non-existent column
        with self.assertRaises(ValueError):
            self.data_smells.check_date_as_datetime(df_dates, 'non_existent')
        print_and_log("Test Case 7 Passed: ValueError raised for non-existent column")

        # Test 8: Test with multiple datetime columns
        df_multiple = pd.DataFrame({
            'dates_only': pd.date_range('2024-01-01', periods=3, freq='D'),
            'with_times': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-02 15:45:30'),
                pd.Timestamp('2024-01-03 08:20:15')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_multiple)
        assert result is False, "Test Case 8 Failed: Should detect smell in at least one column"
        print_and_log("Test Case 8 Passed: Smell detected in multiple columns check")

        # Test 9: Test with single timestamp at exact midnight
        df_single_midnight = pd.DataFrame({
            'single_midnight': [pd.Timestamp('2024-01-01 00:00:00')]
        })
        result = self.data_smells.check_date_as_datetime(df_single_midnight, 'single_midnight')
        assert result is False, "Test Case 9 Failed: Should detect smell for single midnight timestamp"
        print_and_log("Test Case 9 Passed: Smell detected for single midnight timestamp")

        # Test 10: Test with timestamps all at different times
        df_different_times = pd.DataFrame({
            'different_times': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-01 15:45:30'),
                pd.Timestamp('2024-01-01 23:59:59')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_different_times, 'different_times')
        assert result is True, "Test Case 10 Failed: Should not detect smell for different times"
        print_and_log("Test Case 10 Passed: No smell detected for different times")

        # Test 11: Test with timezone-aware datetimes
        df_timezone = pd.DataFrame({
            'timezone_dates': pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
        })
        result = self.data_smells.check_date_as_datetime(df_timezone, 'timezone_dates')
        assert result is False, "Test Case 11 Failed: Should detect smell for timezone-aware dates"
        print_and_log("Test Case 11 Passed: Smell detected for timezone-aware dates")

        # Test 12: Test with microsecond precision
        df_microseconds = pd.DataFrame({
            'with_microseconds': [
                pd.Timestamp('2024-01-01 00:00:00.000001'),
                pd.Timestamp('2024-01-02 00:00:00.000001')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_microseconds, 'with_microseconds')
        assert result is True, "Test Case 12 Failed: Should not detect smell with microseconds"
        print_and_log("Test Case 12 Passed: No smell detected with microseconds")

        # Test 13: Test with end-of-day timestamps
        df_end_of_day = pd.DataFrame({
            'end_of_day': [
                pd.Timestamp('2024-01-01 23:59:59'),
                pd.Timestamp('2024-01-02 23:59:59')
            ]
        })
        result = self.data_smells.check_date_as_datetime(df_end_of_day, 'end_of_day')
        assert result is True, "Test Case 13 Failed: Should not detect smell for end-of-day times"
        print_and_log("Test Case 13 Passed: No smell detected for end-of-day times")

        # Test 14: Test with leap year dates
        df_leap_year = pd.DataFrame({
            'leap_year': pd.date_range('2024-02-28', '2024-03-01', freq='D')
        })
        result = self.data_smells.check_date_as_datetime(df_leap_year, 'leap_year')
        assert result is False, "Test Case 14 Failed: Should detect smell for leap year dates"
        print_and_log("Test Case 14 Passed: Smell detected for leap year dates")

        print_and_log("\nFinished testing check_date_as_datetime function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_separating_consistency_SimpleTests(self):
        """
        Execute simple tests for check_separating_consistency function.
        Tests various scenarios with different decimal and thousands separators.
        """
        print_and_log("")
        print_and_log("Testing check_separating_consistency function...")

        # Test data with various separator cases
        data = {
            'correct_format': [1234.56, 2345.67, 3456.78],
            'wrong_decimal': ['1234,56', '2345,67', '3456,78'],
            'mixed_decimal': ['1234.56', '2345,67', '3456.78'],
            'with_thousands': ['1,234.56', '2,345.67', '3,456.78'],
            'true_thousands': ['1.234,56', '2.345,67', '3.456,78'],
            'mixed_separators': ['1,234.56', '2.345,67', '3,456.78'],
            'no_decimal': [1234, 2345, 3456],
            'scientific': [1.234e3, 2.345e3, 3.456e3],
            'negative': [-1234.56, -2345.67, -3456.78],
            'zero_values': [0.00, 0.0, 0],
            'large_numbers': [1234567.89, 2345678.90, 3456789.01],
            'small_decimals': [0.0001, 0.0002, 0.0003],
            'wrong_grouping': ['1,23,456.78', '2,34,567.89', '3,45,678.90'],
            'non_numeric': ['abc', 'def', 'ghi'],
            'mixed_types': [1234.56, '2,345.67', 3456.78]
        }
        df = pd.DataFrame(data)

        # Test 1: Default separators (decimal=".", thousands="")
        result = self.data_smells.check_separating_consistency(df, ".", "", 'correct_format')
        assert result is True, "Test Case 1 Failed"
        print_and_log("Test Case 1 Passed: Default separators check successful")

        # Test 2: Wrong decimal separator
        result = self.data_smells.check_separating_consistency(df, ".", "", 'wrong_decimal')
        assert result is False, "Test Case 2 Failed"
        print_and_log("Test Case 2 Passed: Wrong decimal separator detected")

        # Test 3: Mixed decimal separators
        result = self.data_smells.check_separating_consistency(df, ".", "", 'mixed_decimal')
        assert result is False, "Test Case 3 Failed"
        print_and_log("Test Case 3 Passed: Mixed decimal separators detected")

        # Test 4: With thousands separator
        result = self.data_smells.check_separating_consistency(df, ".", ",", 'with_thousands')
        assert result is True, "Test Case 4 Failed"
        print_and_log("Test Case 4 Passed: Thousands separator check successful")

        # Test 5: Wrong thousands separator
        result = self.data_smells.check_separating_consistency(df, ",", ".", 'true_thousands')
        assert result is True, "Test Case 5 Failed"
        print_and_log("Test Case 5 Passed: True thousands separator check successful")

        # Test 6: Mixed separators
        result = self.data_smells.check_separating_consistency(df, ".", ",", 'mixed_separators')
        assert result is False, "Test Case 6 Failed"
        print_and_log("Test Case 6 Passed: Mixed separators detected")

        # Test 7: No decimal values
        result = self.data_smells.check_separating_consistency(df, ".", ",", 'no_decimal')
        assert result is True, "Test Case 7 Failed"
        print_and_log("Test Case 7 Passed: No decimal values check successful")

        # Test 8: Scientific notation
        result = self.data_smells.check_separating_consistency(df, ".", "", 'scientific')
        assert result is True, "Test Case 8 Failed"
        print_and_log("Test Case 8 Passed: Scientific notation check successful")

        # Test 9: Negative numbers
        result = self.data_smells.check_separating_consistency(df, ".", "", 'negative')
        assert result is True, "Test Case 9 Failed"
        print_and_log("Test Case 9 Passed: Negative numbers check successful")

        # Test 10: Zero values
        result = self.data_smells.check_separating_consistency(df, ".", "", 'zero_values')
        assert result is True, "Test Case 10 Failed"
        print_and_log("Test Case 10 Passed: Zero values check successful")

        # Test 11: Large numbers
        result = self.data_smells.check_separating_consistency(df, ".", "", 'large_numbers')
        assert result is True, "Test Case 11 Failed"
        print_and_log("Test Case 11 Passed: Large numbers check successful")

        # Test 12: Small decimals
        result = self.data_smells.check_separating_consistency(df, ".", "", 'small_decimals')
        assert result is True, "Test Case 12 Failed"
        print_and_log("Test Case 12 Passed: Small decimals check successful")

        # Test 13: Wrong grouping with thousands separator
        result = self.data_smells.check_separating_consistency(df, ".", ",", 'wrong_grouping')
        assert result is False, "Test Case 13 Failed"
        print_and_log("Test Case 13 Passed: Wrong grouping detected")

        # Test 14: Non-numeric column
        result = self.data_smells.check_separating_consistency(df, ".", "", 'non_numeric')
        assert result is True, "Test Case 14 Failed"
        print_and_log("Test Case 14 Passed: Non-numeric column check successful")

        # Test 15: Mixed types
        result = self.data_smells.check_separating_consistency(df, ".", "", 'mixed_types')
        assert result is False, "Test Case 15 Failed"
        print_and_log("Test Case 15 Passed: Mixed types detected")

        print_and_log("\nFinished testing check_separating_consistency function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_date_time_consistency_SimpleTests(self):
        """
        Execute simple tests for check_date_time_consistency function.
        Tests the following cases:
        1. Pure dates in Date type column (no smell)
        2. Dates with time in Date type column (smell)
        3. Mixed dates in DateTime type column (no smell)
        4. Non-datetime column (no smell)
        5. Empty DataFrame
        6. Column with NaN values
        7. Non-existent column
        8. Invalid DataType
        9. Column with timezone aware dates
        10. Column with microsecond precision
        11. Column with mixed timezones
        12. Column with only midnight times
        13. Single date value
        14. All columns check
        15. Dates at different times of day
        """
        print_and_log("")
        print_and_log("Testing check_date_time_consistency function...")

        # Test 1: Pure dates in Date type column (no smell)
        df_pure_dates = pd.DataFrame({
            'pure_dates': pd.date_range('2024-01-01', periods=3, freq='D')
        })
        result = self.data_smells.check_date_time_consistency(df_pure_dates, DataType.DATE, 'pure_dates')
        assert result is True, "Test Case 1 Failed: Should not detect smell for pure dates with Date type"
        print_and_log("Test Case 1 Passed: No smell detected for pure dates with Date type")

        # Test  2: Dates with time in Date type column (smell)
        df_dates_with_time = pd.DataFrame({
            'dates_with_time': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-02 15:45:30'),
                pd.Timestamp('2024-01-03 08:20:15')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_dates_with_time, DataType.DATE, 'dates_with_time')
        assert result is False, "Test Case 2 Failed: Should detect smell for dates with time in Date type"
        print_and_log("Test Case 2 Passed: Smell detected for dates with time in Date type")

        # Test 3: Mixed dates in DateTime type column (no smell)
        df_datetime = pd.DataFrame({
            'datetime_col': [
                pd.Timestamp('2024-01-01 10:30:00'),
                pd.Timestamp('2024-01-02 00:00:00'),
                pd.Timestamp('2024-01-03 23:59:59')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_datetime, DataType.DATETIME, 'datetime_col')
        assert result is True, "Test Case 3 Failed: Should not detect smell for mixed times in DateTime type"
        print_and_log("Test Case 3 Passed: No smell detected for mixed times in DateTime type")

        # Test 4: Non-datetime column (no smell)
        df_non_datetime = pd.DataFrame({
            'strings': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        result = self.data_smells.check_date_time_consistency(df_non_datetime, DataType.DATE, 'strings')
        assert result is True, "Test Case 4 Failed: Should not detect smell for non-datetime column"
        print_and_log("Test Case 4 Passed: No smell detected for non-datetime column")

        # Test 5: Empty DataFrame
        df_empty = pd.DataFrame()
        result = self.data_smells.check_date_time_consistency(df_empty, DataType.DATE)
        assert result is True, "Test Case 5 Failed: Should not detect smell for empty DataFrame"
        print_and_log("Test Case 5 Passed: No smell detected for empty DataFrame")

        # Test 6: Column with NaN values
        df_with_nan = pd.DataFrame({
            'dates_with_nan': [pd.Timestamp('2024-01-01'), np.nan, pd.Timestamp('2024-01-03')]
        })
        result = self.data_smells.check_date_time_consistency(df_with_nan, DataType.DATE, 'dates_with_nan')
        assert result is True, "Test Case 6 Failed: Should not detect smell for dates with NaN"
        print_and_log("Test Case 6 Passed: No smell detected for dates with NaN")

        # Test 7: Non-existent column
        with self.assertRaises(ValueError):
            self.data_smells.check_date_time_consistency(df_pure_dates, 'non_existent')
        print_and_log("Test Case 7 Passed: ValueError raised for non-existent column")

        # Test 8: Invalid DataType
        with self.assertRaises(ValueError):
            self.data_smells.check_date_time_consistency(df_datetime, DataType.STRING, 'datetime_col')
        print_and_log("Test Case 8 Passed: ValueError raised for invalid DataType")

        # Test 9: Column with timezone aware dates
        df_timezone = pd.DataFrame({
            'timezone_dates': pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
        })
        result = self.data_smells.check_date_time_consistency(df_timezone, DataType.DATE, 'timezone_dates')
        assert result is True, "Test Case 9 Failed: Should not detect smell for timezone-aware dates"
        print_and_log("Test Case 9 Passed: No smell detected for timezone-aware dates")

        # Test 10: Column with microsecond precision
        df_microseconds = pd.DataFrame({
            'microseconds': [pd.Timestamp('2024-01-01 00:00:00.000001')]
        })
        result = self.data_smells.check_date_time_consistency(df_microseconds, DataType.DATE, 'microseconds')
        assert result is False, "Test Case 10 Failed: Should detect smell for dates with microseconds"
        print_and_log("Test Case 10 Passed: Smell detected for dates with microseconds")

        # Test 11: Column with mixed timezones
        df_mixed_tz = pd.DataFrame({
            'mixed_tz': [
                pd.Timestamp('2024-01-01', tz='UTC'),
                pd.Timestamp('2024-01-02', tz='US/Eastern'),
                pd.Timestamp('2024-01-03', tz='Europe/London')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_mixed_tz, DataType.DATE, 'mixed_tz')
        assert result is True, "Test Case 11 Failed: Should not detect smell for dates with mixed timezones"
        print_and_log("Test Case 11 Passed: No smell detected for dates with mixed timezones")

        # Test 12: Column with only midnight times
        df_midnight = pd.DataFrame({
            'midnight_times': [
                pd.Timestamp('2024-01-01 00:00:00'),
                pd.Timestamp('2024-01-02 00:00:00'),
                pd.Timestamp('2024-01-03 00:00:00')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_midnight, DataType.DATE, 'midnight_times')
        assert result is True, "Test Case 12 Failed: Should not detect smell for midnight times"
        print_and_log("Test Case 12 Passed: No smell detected for midnight times")

        # Test 13: Single date value
        df_single = pd.DataFrame({
            'single_date': [pd.Timestamp('2024-01-01')]
        })
        result = self.data_smells.check_date_time_consistency(df_single, DataType.DATE, 'single_date')
        assert result is True, "Test Case 13 Failed: Should not detect smell for single date"
        print_and_log("Test Case 13 Passed: No smell detected for single date")

        # Test 14: All columns check
        df_multiple = pd.DataFrame({
            'dates1': pd.date_range('2024-01-01', periods=3, freq='D'),
            'dates2': [pd.Timestamp('2024-01-01 10:30:00')] * 3
        })
        result = self.data_smells.check_date_time_consistency(df_multiple, DataType.DATE)
        assert result is False, "Test Case 14 Failed: Should detect smell in at least one column"
        print_and_log("Test Case 14 Passed: Smell detected in multiple columns check")

        # Test 15: Dates at different times of day
        df_times = pd.DataFrame({
            'different_times': [
                pd.Timestamp('2024-01-01 09:00:00'),
                pd.Timestamp('2024-01-01 12:00:00'),
                pd.Timestamp('2024-01-01 17:00:00')
            ]
        })
        result = self.data_smells.check_date_time_consistency(df_times, DataType.DATE, 'different_times')
        assert result is False, "Test Case 15 Failed: Should detect smell for different times of day"
        print_and_log("Test Case 15 Passed: Smell detected for different times of day")

        print_and_log("\nFinished testing check_date_time_consistency function")
        print_and_log("-----------------------------------------------------------")

    def execute_check_ambiguous_datetime_format_SimpleTests(self):
        """
        Execute simple tests for check_ambiguous_datetime_format function.
        Tests detection of the specific %I:%M %p pattern (HH:MM AM/PM).
        """
        print_and_log("")
        print_and_log("Testing check_ambiguous_datetime_format function...")

        # Test 1: DateTime values with HH:MM AM/PM pattern (smell)
        df1 = pd.DataFrame({'datetime_col': ['2025-06-26 02:30 PM', '2025-06-26 09:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df1, 'datetime_col')
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for HH:MM AM/PM pattern")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: 24-hour format values (no smell)
        df2 = pd.DataFrame({'datetime_col': ['2025-06-26 14:30:00', '2025-06-26 09:15:30']})
        result = self.data_smells.check_ambiguous_datetime_format(df2, 'datetime_col')
        self.assertTrue(result, "Test Case 2 Failed: Expected no smell for 24-hour format")
        print_and_log("Test Case 2 Passed: Expected no smell, got no smell")

        # Test 3: Simple time values with HH:MM AM/PM (smell)
        df3 = pd.DataFrame({'time_col': ['02:30 PM', '09:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df3, 'time_col')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for HH:MM AM/PM pattern")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Date-only values (no smell)
        df4 = pd.DataFrame({'date_col': ['2025-06-26', '2025-12-31']})
        result = self.data_smells.check_ambiguous_datetime_format(df4, 'date_col')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for date-only values")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Non-existent field
        with self.assertRaises(ValueError):
            self.data_smells.check_ambiguous_datetime_format(df1, 'non_existent_field')
        print_and_log("Test Case 5 Passed: Expected ValueError for non-existent field")

        # Test 6: Time with seconds and AM/PM (smell - contains AM/PM indicators)
        df6 = pd.DataFrame({'datetime_col': ['02:30:45 PM', '09:15:30 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df6, 'datetime_col')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for times with AM/PM indicators")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: 24-hour time values (no smell)
        df7 = pd.DataFrame({'time_col': ['14:30', '09:15']})
        result = self.data_smells.check_ambiguous_datetime_format(df7, 'time_col')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for 24-hour time")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Mixed case AM/PM with HH:MM pattern (smell)
        df8 = pd.DataFrame({'time_col': ['02:30 pm', '09:15 Am']})
        result = self.data_smells.check_ambiguous_datetime_format(df8, 'time_col')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for HH:MM am/pm pattern")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Empty DataFrame (no smell)
        df9 = pd.DataFrame({'datetime_col': []})
        result = self.data_smells.check_ambiguous_datetime_format(df9, 'datetime_col')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for empty DataFrame")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: 12-hour format with 12:XX times (smell)
        df10 = pd.DataFrame({'time_col': ['12:30 PM', '12:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df10, 'time_col')
        self.assertFalse(result, "Test Case 10 Failed: Expected smell for 12:XX AM/PM pattern")
        print_and_log("Test Case 10 Passed: Expected smell, got smell")

        # Test 11: Single digit hours with HH:MM AM/PM (smell)
        df11 = pd.DataFrame({'time_col': ['1:30 PM', '9:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df11, 'time_col')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for H:MM AM/PM pattern")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Dotted AM/PM format with HH:MM (smell)
        df12 = pd.DataFrame({'time_col': ['02:30 a.m.', '09:15 p.m.']})
        result = self.data_smells.check_ambiguous_datetime_format(df12, 'time_col')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for HH:MM a.m./p.m. pattern")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Text with AM/PM but no time pattern (smell - contains AM/PM indicators)
        df13 = pd.DataFrame({'text_col': ['The meeting is in the AM session', 'PM responsibilities']})
        result = self.data_smells.check_ambiguous_datetime_format(df13, 'text_col')
        self.assertFalse(result, "Test Case 13 Failed: Expected smell for text with AM/PM indicators")
        print_and_log("Test Case 13 Passed: Expected smell, got smell")

        # Test 14: Invalid time hours (13-23) with AM/PM (smell - contains AM/PM indicators)
        df14 = pd.DataFrame({'time_col': ['14:30 PM', '23:15 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df14, 'time_col')
        self.assertFalse(result, "Test Case 14 Failed: Expected smell for times with AM/PM indicators")
        print_and_log("Test Case 14 Passed: Expected smell, got smell")

        # Test 15: Complex datetime with HH:MM AM/PM pattern (smell)
        df15 = pd.DataFrame({'datetime_col': ['Monday, June 26, 2025 at 2:30 PM']})
        result = self.data_smells.check_ambiguous_datetime_format(df15, 'datetime_col')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for complex datetime with H:MM AM/PM")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

        # Test 16: Time ranges with HH:MM AM/PM (smell)
        df16 = pd.DataFrame({'time_col': ['02:30 PM - 03:45 PM', '09:15 AM - 10:30 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df16, 'time_col')
        self.assertFalse(result, "Test Case 16 Failed: Expected smell for time ranges with HH:MM AM/PM")
        print_and_log("Test Case 16 Passed: Expected smell, got smell")

        # Test 17: Numbers that look like times but no AM/PM (no smell)
        df17 = pd.DataFrame({'time_col': ['1430', '0915', '02:30']})
        result = self.data_smells.check_ambiguous_datetime_format(df17, 'time_col')
        self.assertTrue(result, "Test Case 17 Failed: Expected no smell for times without AM/PM")
        print_and_log("Test Case 17 Passed: Expected no smell, got no smell")

        # Test 18: Invalid minutes (>59) with AM/PM (smell - contains AM/PM indicators)
        df18 = pd.DataFrame({'time_col': ['02:75 PM', '09:99 AM']})
        result = self.data_smells.check_ambiguous_datetime_format(df18, 'time_col')
        self.assertFalse(result, "Test Case 18 Failed: Expected smell for times with AM/PM indicators")
        print_and_log("Test Case 18 Passed: Expected smell, got smell")

        # Test 19: Mixed formats with some HH:MM AM/PM (smell detected)
        df19 = pd.DataFrame({'datetime_col': ['2025-06-26 14:30:00', '02:30 PM', '2025-06-26 16:45:00']})
        result = self.data_smells.check_ambiguous_datetime_format(df19, 'datetime_col')
        self.assertFalse(result, "Test Case 19 Failed: Expected smell for mixed formats with HH:MM AM/PM")
        print_and_log("Test Case 19 Passed: Expected smell, got smell")

        # Test 20: Null values only (no smell)
        df20 = pd.DataFrame({'datetime_col': [None, np.nan, pd.NaT]})
        result = self.data_smells.check_ambiguous_datetime_format(df20, 'datetime_col')
        self.assertTrue(result, "Test Case 20 Failed: Expected no smell for null values only")
        print_and_log("Test Case 20 Passed: Expected no smell, got no smell")

        print_and_log("\nFinished testing check_ambiguous_datetime_format function")

    def execute_check_suspect_date_value_SimpleTests(self):
        """
        Execute simple tests for check_suspect_date_value function.
        Tests the following cases:
        1. Invalid date format in parameters
        2. min_date greater than max_date
        3. Non-existent field
        4. Valid dates within range (no smell)
        5. Dates outside range (smell detected)
        6. Mixed dates - some in range, some out of range (smell detected)
        7. Empty DataFrame
        8. Column with all NaN/NaT values
        9. Non-datetime column (should pass)
        10. Object column with date strings
        11. Timezone-aware datetime column
        12. Check all datetime columns at once
        """
        print_and_log("")
        print_and_log("Testing check_suspect_date_value function...")

        # Test 1: Invalid date format in parameters
        df1 = pd.DataFrame({'date_col': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31'])})
        try:
            result = self.data_smells.check_suspect_date_value(df1, 'invalid-date', '2023-12-31')
            self.fail("Test Case 1 Failed: Expected ValueError for invalid date format")
        except ValueError:
            print_and_log("Test Case 1 Passed: Expected ValueError for invalid date format")

        # Test 2: min_date greater than max_date
        try:
            result = self.data_smells.check_suspect_date_value(df1, '2023-12-31', '2023-01-01')
            self.fail("Test Case 2 Failed: Expected ValueError for min_date > max_date")
        except ValueError:
            print_and_log("Test Case 2 Passed: Expected ValueError for min_date > max_date")

        # Test 3: Non-existent field
        try:
            result = self.data_smells.check_suspect_date_value(df1, '2023-01-01', '2023-12-31', 'non_existent')
            self.fail("Test Case 3 Failed: Expected ValueError for non-existent field")
        except ValueError:
            print_and_log("Test Case 3 Passed: Expected ValueError for non-existent field")

        # Test 4: Valid dates within range (no smell)
        df4 = pd.DataFrame({'date_col': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31'])})
        result = self.data_smells.check_suspect_date_value(df4, '2022-01-01', '2024-01-01', 'date_col')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for dates within range")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Dates outside range (smell detected)
        df5 = pd.DataFrame({'date_col': pd.to_datetime(['2021-01-01', '2023-06-15', '2025-12-31'])})
        result = self.data_smells.check_suspect_date_value(df5, '2022-01-01', '2024-01-01', 'date_col')
        self.assertFalse(result, "Test Case 5 Failed: Expected smell for dates outside range")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        # Test 6: Mixed dates - some in range, some out of range (smell detected)
        df6 = pd.DataFrame({'date_col': pd.to_datetime(['2023-01-01', '2025-06-15', '2023-12-31'])})
        result = self.data_smells.check_suspect_date_value(df6, '2023-01-01', '2023-12-31', 'date_col')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for mixed dates")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Empty DataFrame
        df7 = pd.DataFrame()
        result = self.data_smells.check_suspect_date_value(df7, '2023-01-01', '2023-12-31')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for empty DataFrame")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Column with all NaN/NaT values
        df8 = pd.DataFrame({'date_col': pd.to_datetime([None, np.nan, pd.NaT])})
        result = self.data_smells.check_suspect_date_value(df8, '2023-01-01', '2023-12-31', 'date_col')
        self.assertTrue(result, "Test Case 8 Failed: Expected no smell for all NaN/NaT values")
        print_and_log("Test Case 8 Passed: Expected no smell, got no smell")

        # Test 9: Non-datetime column (should pass)
        df9 = pd.DataFrame({'non_date_col': [1, 2, 3, 4, 5]})
        result = self.data_smells.check_suspect_date_value(df9, '2023-01-01', '2023-12-31', 'non_date_col')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for non-datetime column")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: Object column with date strings (should pass - not datetime column)
        df10 = pd.DataFrame({'date_col': ['2023-01-01', '2025-06-15', '2023-12-31']})
        result = self.data_smells.check_suspect_date_value(df10, '2023-01-01', '2023-12-31', 'date_col')
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for object column with date strings (not datetime)")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Timezone-aware datetime column
        df11 = pd.DataFrame({'date_col': pd.to_datetime(['2023-01-01', '2025-06-15', '2023-12-31'], utc=True)})
        result = self.data_smells.check_suspect_date_value(df11, '2023-01-01', '2023-12-31', 'date_col')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for timezone-aware dates outside range")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Check all datetime columns at once
        df12 = pd.DataFrame({
            'date_col1': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31']),
            'date_col2': pd.to_datetime(['2022-01-01', '2023-06-15', '2025-12-31']),
            'non_date_col': [1, 2, 3],
            'string_dates': ['2025-01-01', '2025-06-15', '2025-12-31']  # This should be ignored
        })
        result = self.data_smells.check_suspect_date_value(df12, '2023-01-01', '2023-12-31')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell when checking datetime columns only")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: All datetime columns within range
        df13 = pd.DataFrame({
            'date_col1': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31']),
            'date_col2': pd.to_datetime(['2023-02-01', '2023-07-15', '2023-11-30']),
            'non_date_col': [1, 2, 3],
            'string_dates': ['2025-01-01', '2025-06-15', '2025-12-31']  # This should be ignored
        })
        result = self.data_smells.check_suspect_date_value(df13, '2023-01-01', '2023-12-31')
        self.assertTrue(result, "Test Case 13 Failed: Expected no smell when all datetime columns within range")
        print_and_log("Test Case 13 Passed: Expected no smell, got no smell")

        print_and_log("\nFinished testing check_suspect_date_value function")