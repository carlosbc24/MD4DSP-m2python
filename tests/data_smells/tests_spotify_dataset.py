# Importing libraries
import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm
import functions.data_smells as data_smells
from helpers.enumerations import DataType

# Importing functions and classes from packages
from helpers.logger import print_and_log


class DataSmellExternalDatasetTests(unittest.TestCase):
    """
    Class to test the Data Smells with external dataset test cases

    Attributes:
        data_smells (DataSmells): instance of the class DataSmells
        data_dictionary (pd.DataFrame): dataframe with the external dataset
    """

    def __init__(self):
        """
        Constructor of the class that initializes the data_smells instance and loads the dataset
        """
        super().__init__()
        self.data_smells = data_smells

        # Get the current directory path of the script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the CSV file
        csv_path = os.path.join(current_dir, '../../test_datasets/spotify_songs/spotify_songs.csv')
        # Create DataFrame from CSV file
        self.data_dictionary = pd.read_csv(csv_path)

    def executeAll_ExternalDatasetTests(self):
        """
        Execute all external dataset tests
        """
        test_methods = [
            self.execute_check_precision_consistency_ExternalDatasetTests,
            self.execute_check_missing_invalid_value_consistency_ExternalDatasetTests,
            self.execute_check_integer_as_floating_point_ExternalDatasetTests,
            self.execute_check_types_as_string_ExternalDatasetTests,
            self.execute_check_special_character_spacing_ExternalDatasetTests,
            self.execute_check_suspect_distribution_ExternalDatasetTests,
            self.execute_check_ambiguous_datetime_format_ExternalDatasetTests
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ STARTING DATA-SMELL DATASET TEST CASES -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for test_method in tqdm(test_methods, desc="Running Data Smell External Dataset Tests",
                              unit="test"):
            test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ DATA-SMELL DATASET TEST CASES FINISHED -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")

    def execute_check_precision_consistency_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_precision_consistency function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_precision_consistency Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check danceability field with correct decimal places (3)
        print_and_log("\nTest 1: Check danceability field with correct decimal places")
        result = self.data_smells.check_precision_consistency(test_df, 3, 'danceability')
        assert result is False, "Test Case 1 Failed: Expected False, but got True"
        print_and_log("Test Case 1 Passed: Expected False, got False")

        # Test 2: Check danceability field with incorrect decimal places (4)
        print_and_log("\nTest 2: Check danceability field with incorrect decimal places")
        result = self.data_smells.check_precision_consistency(test_df, 4, 'danceability')
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Test 3: Check energy field with correct decimal places (0)
        print_and_log("\nTest 3: Check energy field with correct decimal places (0), as mode is an integer column")
        result = self.data_smells.check_precision_consistency(test_df, 0, 'mode')
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Test 4: Check non-numeric field (track_name)
        print_and_log("\nTest 4: Check non-numeric field")
        result = self.data_smells.check_precision_consistency(test_df, 3, 'track_name')
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Test 5: Check integer field (key) with 0 decimals
        print_and_log("\nTest 5: Check integer field with 0 decimals")
        result = self.data_smells.check_precision_consistency(test_df, 0, 'key')
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Test 6: Add inconsistent decimals to a copy of loudness field
        print_and_log("\nTest 6: Check field with inconsistent decimals")
        test_df['loudness_test'] = test_df['loudness'].apply(lambda x: round(x, np.random.choice([2, 3, 4])))
        result = self.data_smells.check_precision_consistency(test_df, 3, 'loudness_test')
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Test 7: Check all numeric fields at once with expected 3 decimals
        print_and_log("\nTest 7: Check all numeric fields at once")
        result = self.data_smells.check_precision_consistency(test_df, 3, None)
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Test 8: Check field with null values
        print_and_log("\nTest 8: Check field with null values")
        test_df['test_nulls'] = test_df['loudness'].copy()
        test_df.loc[0:10, 'test_nulls'] = np.nan
        result = self.data_smells.check_precision_consistency(test_df, 3, 'test_nulls')
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Test 9: Check with negative expected decimals
        print_and_log("\nTest 9: Check with negative expected decimals")
        with self.assertRaises(TypeError):
            self.data_smells.check_precision_consistency(test_df, -1, 'danceability')
        print_and_log("Test Case 9 Passed: Expected TypeError, got TypeError")

        # Test 10: Check non-existent field
        print_and_log("\nTest 10: Check non-existent field")
        with self.assertRaises(ValueError):
            self.data_smells.check_precision_consistency(test_df, 3, 'non_existent_field')
        print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Test 11: Check valence field with correct decimals
        print_and_log("\nTest 11: Check valence field with correct decimals")
        result = self.data_smells.check_precision_consistency(test_df, 3, 'valence')
        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Test 12: Check key field with expected decimals
        print_and_log("\nTest 12: Check key field")
        result = self.data_smells.check_precision_consistency(test_df, 0, 'key')
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Test 13: Check field with some zero values
        print_and_log("\nTest 13: Check field with zero values")
        test_df['test_zeros'] = test_df['loudness'].copy()
        test_df.loc[0:10, 'test_zeros'] = 0.000
        result = self.data_smells.check_precision_consistency(test_df, 3, 'test_zeros')
        assert result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, got False")

        # Test 14: Check empty DataFrame
        print_and_log("\nTest 14: Check empty DataFrame")
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.data_smells.check_precision_consistency(empty_df, 3, 'any_field')
        print_and_log("Test Case 14 Passed: Expected ValueError, got ValueError")

        # Test 15: Check field with mixed integer and float values
        print_and_log("\nTest 15: Check field with mixed integer and float values")
        test_df['mixed_values'] = test_df['loudness'].apply(lambda x: int(x) if x % 2 == 0 else x)
        result = self.data_smells.check_precision_consistency(test_df, 3, 'mixed_values')
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Test 16: Check acousticness field
        print_and_log("\nTest 16: Check acousticness field")
        result = self.data_smells.check_precision_consistency(test_df, 3, 'acousticness')
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Test 17: Check speechiness field
        print_and_log("\nTest 17: Check speechiness field")
        result = self.data_smells.check_precision_consistency(test_df, 7, 'speechiness')
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Test 18: Check liveness field
        print_and_log("\nTest 18: Check liveness field")
        result = self.data_smells.check_precision_consistency(test_df, 8, 'liveness')
        assert result is False, "Test Case 18 Failed: Expected False, but got True"
        print_and_log("Test Case 18 Passed: Expected False, got False")

        # Test 19: Check tempo field
        print_and_log("\nTest 19: Check tempo field")
        result = self.data_smells.check_precision_consistency(test_df, 0, 'duration_ms')
        assert result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")

        # Test 20: Check duration_ms field (integer field) with non-zero decimals
        print_and_log("\nTest 20: Check integer field with non-zero decimals")
        result = self.data_smells.check_precision_consistency(test_df, 2, 'duration_ms')
        assert result is False, "Test Case 20 Failed: Expected False, but got True"
        print_and_log("Test Case 20 Passed: Expected False, got False")

        print_and_log("\nFinished testing check_precision_consistency function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_missing_invalid_value_consistency_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_missing_invalid_value_consistency function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_missing_invalid_value_consistency Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Common test values
        invalid_common = ['inf', '-inf', 'nan']
        missing_common = ['', '?', '.', 'null', 'none', 'na']

        # Test 1: Check track_name column (should be clean)
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, 'track_name')
        assert result is False, "Test Case 1 Failed"
        print_and_log("Test Case 1 Passed: track_name column check successful")

        # Test 2: Modify track_name to include missing values
        test_df.loc[0:10, 'track_name'] = 'na'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, 'track_name')
        assert result is False, "Test Case 2 Failed"
        print_and_log("Test Case 2 Passed: Modified track_name with missing values detected")

        # Test 3: Check danceability column with invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'danceability')
        assert result is True, "Test Case 3 Failed"
        print_and_log("Test Case 3 Passed: danceability column check successful")

        # Test 4: Modify danceability to include invalid values
        test_df.loc[0:5, 'danceability'] = 'inf'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'danceability')
        assert result is False, "Test Case 4 Failed"
        print_and_log("Test Case 4 Passed: Modified danceability with invalid values detected")

        # Test 5: Check energy column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'energy')
        assert result is True, "Test Case 5 Failed"
        print_and_log("Test Case 5 Passed: energy column check successful")

        # Test 6: Check key column with custom missing values
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, ['-1'], missing_common, 'key')
        assert result is True, "Test Case 6 Failed"
        print_and_log("Test Case 6 Passed: key column with custom missing values check successful")

        # Test 7: Check loudness column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'loudness')
        assert result is True, "Test Case 7 Failed"
        print_and_log("Test Case 7 Passed: loudness column check successful")

        # Test 8: Modify loudness to include invalid values
        test_df.loc[0:5, 'loudness'] = 'nan'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'loudness')
        assert result is False, "Test Case 8 Failed"
        print_and_log("Test Case 8 Passed: Modified loudness with invalid values detected")

        # Test 9: Check mode column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, 'mode')
        assert result is True, "Test Case 9 Failed"
        print_and_log("Test Case 9 Passed: mode column check successful")

        # Test 10: Check speechiness column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'speechiness')
        assert result is True, "Test Case 10 Failed"
        print_and_log("Test Case 10 Passed: speechiness column check successful")

        # Test 11: Check acousticness with mixed invalid values
        test_df.loc[0:3, 'acousticness'] = 'inf'
        test_df.loc[4:7, 'acousticness'] = '-inf'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'acousticness')
        assert result is False, "Test Case 11 Failed"
        print_and_log("Test Case 11 Passed: acousticness with mixed invalid values detected")

        # Test 12: Check instrumentalness column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'instrumentalness')
        assert result is True, "Test Case 12 Failed"
        print_and_log("Test Case 12 Passed: instrumentalness column check successful")

        # Test 13: Check liveness column with custom invalid values
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, ['999'], invalid_common, 'liveness')
        assert result is True, "Test Case 13 Failed"
        print_and_log("Test Case 13 Passed: liveness with custom invalid values check successful")

        # Test 14: Check valence column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'valence')
        assert result is True, "Test Case 14 Failed"
        print_and_log("Test Case 14 Passed: valence column check successful")

        # Test 15: Check tempo column with modified values
        test_df.loc[0:5, 'tempo'] = 'null'
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, 'tempo')
        assert result is False, "Test Case 15 Failed"
        print_and_log("Test Case 15 Passed: Modified tempo with missing values detected")

        # Test 16: Check duration_ms column
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, 'duration_ms')
        assert result is True, "Test Case 16 Failed"
        print_and_log("Test Case 16 Passed: duration_ms column check successful")

        # Test 17: Check all numeric columns at once
        numeric_columns = test_df.select_dtypes(include=['float64', 'Int64']).columns
        all_valid = all(self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], invalid_common, col) for col in numeric_columns)
        assert all_valid, "Test Case 17 Failed"
        print_and_log("Test Case 17 Passed: All numeric columns check successful")

        # Test 18: Check all string columns with missing values
        string_columns = test_df.select_dtypes(include=['object']).columns
        all_valid = all(self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common, col) for col in string_columns)
        assert not all_valid, "Test Case 18 Failed"
        print_and_log("Test Case 18 Passed: All string columns check successful")

        # Test 19: Check with empty model definitions
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], [], 'track_name')
        assert result is True, "Test Case 19 Failed"
        print_and_log("Test Case 19 Passed: Empty model definitions check successful")

        # Test 20: Check with all columns at once
        result = self.data_smells.check_missing_invalid_value_consistency(
            test_df, [], missing_common + invalid_common)
        assert result is False, "Test Case 20 Failed"
        print_and_log("Test Case 20 Passed: All columns simultaneous check successful")

        print_and_log("\nFinished testing check_missing_invalid_value_consistency function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")

    def execute_check_integer_as_floating_point_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_integer_as_floating_point function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_integer_as_floating_point Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check a genuine float field (danceability)
        print_and_log("\nTest 1: Check a genuine float field (danceability)")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'danceability')
        self.assertTrue(result, "Test Case 1 Failed: Expected no smell for a float column")
        print_and_log("Test Case 1 Passed: Expected no smell, got no smell")

        # Test 2: Check a genuine integer field (duration_ms)
        print_and_log("\nTest 2: Check a genuine integer field (duration_ms)")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'duration_ms')
        self.assertFalse(result, "Test Case 2 Failed: Expected a smell for an integer column stored as float")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Create an integer-as-float column and check for the smell
        print_and_log("\nTest 3: Check an integer-as-float column")
        test_df['duration_ms_float'] = test_df['duration_ms'].astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df, 'duration_ms_float')
        self.assertFalse(result, "Test Case 3 Failed: Expected a smell for an integer-as-float column")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Check a non-numeric field (track_name)
        print_and_log("\nTest 4: Check a non-numeric field (track_name)")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'track_name')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for a non-numeric column")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Check a column with NaNs and integers-as-floats
        print_and_log("\nTest 5: Check a column with NaNs and integers-as-floats")
        test_df['int_as_float_with_nan'] = test_df['duration_ms'].astype(float)
        test_df.loc[0:10, 'int_as_float_with_nan'] = np.nan
        result = self.data_smells.check_integer_as_floating_point(test_df, 'int_as_float_with_nan')
        self.assertFalse(result, "Test Case 5 Failed: Expected a smell for a column with NaNs and int-as-float")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        # Test 6: Check track_popularity column (should be integer stored as float, smell expected)
        print_and_log("\nTest 6: Check track_popularity column")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'track_popularity')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for track_popularity (integers as float)")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Create a column with mixed int and float values (no smell)
        print_and_log("\nTest 7: Check mixed int/float column")
        test_df['mixed_values'] = test_df['danceability'].copy()
        test_df.loc[0:100, 'mixed_values'] = test_df.loc[0:100, 'duration_ms'].astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df, 'mixed_values')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for mixed int/float column")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Check a column with all zero values as float (smell)
        print_and_log("\nTest 8: Check column with all zeros as float")
        test_df['all_zeros_float'] = 0.0
        result = self.data_smells.check_integer_as_floating_point(test_df, 'all_zeros_float')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for all zeros as float")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Check key column (should be integer stored as float if converted)
        print_and_log("\nTest 9: Check key column converted to float")
        test_df['key_as_float'] = test_df['key'].astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df, 'key_as_float')
        self.assertFalse(result, "Test Case 9 Failed: Expected smell for key column as float")
        print_and_log("Test Case 9 Passed: Expected smell, got smell")

        # Test 10: Check mode column converted to float (smell)
        print_and_log("\nTest 10: Check mode column converted to float")
        test_df['mode_as_float'] = test_df['mode'].astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df, 'mode_as_float')
        self.assertFalse(result, "Test Case 10 Failed: Expected smell for mode column as float")
        print_and_log("Test Case 10 Passed: Expected smell, got smell")

        # Test 11: Check a subset of data with integers only (smell)
        print_and_log("\nTest 11: Check subset with integers only")
        test_df_subset = test_df.head(100).copy()
        test_df_subset['tempo_int_as_float'] = test_df_subset['tempo'].round().astype(float)
        result = self.data_smells.check_integer_as_floating_point(test_df_subset, 'tempo_int_as_float')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for rounded tempo as float")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Check energy column (should have decimals, no smell)
        print_and_log("\nTest 12: Check energy column")
        result = self.data_smells.check_integer_as_floating_point(test_df, 'energy')
        self.assertTrue(result, "Test Case 12 Failed: Expected no smell for energy column")
        print_and_log("Test Case 12 Passed: Expected no smell, got no smell")

        # Test 13: Check all columns at once (should find smells)
        print_and_log("\nTest 13: Check all columns at once")
        result = self.data_smells.check_integer_as_floating_point(test_df)
        self.assertFalse(result, "Test Case 13 Failed: Expected smell when checking all columns")
        print_and_log("Test Case 13 Passed: Expected smell when checking all columns, got smell")

        # Test 14: Check with negative values as float (smell)
        print_and_log("\nTest 14: Check negative integers as float")
        # Create a small test DataFrame for this specific test
        small_test_df = pd.DataFrame({
            'negative_int_as_float': [-1.0, -2.0, -3.0, -10.0, -100.0]
        })
        result = self.data_smells.check_integer_as_floating_point(small_test_df, 'negative_int_as_float')
        self.assertFalse(result, "Test Case 14 Failed: Expected smell for negative integers as float")
        print_and_log("Test Case 14 Passed: Expected smell, got smell")

        # Test 15: Check large integer values as float (smell)
        print_and_log("\nTest 15: Check large integers as float")
        # Create a small test DataFrame for this specific test
        large_test_df = pd.DataFrame({
            'large_int_as_float': [1000000.0, 2000000.0, 3000000.0, 4000000.0, 5000000.0]
        })
        result = self.data_smells.check_integer_as_floating_point(large_test_df, 'large_int_as_float')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for large integers as float")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

    def execute_check_types_as_string_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_types_as_string function using the Spotify dataset.
        Tests the following cases:
        1. All values in a column are integer strings (should warn)
        2. All values in a column are float strings (should warn)
        3. All values in a column are date strings (should warn)
        4. Mixed string values (should not warn)
        5. Type mismatch (should raise TypeError)
        6. Non-existent field (should raise ValueError)
        7. Unknown expected_type (should raise ValueError)
        """
        print_and_log("")
        print_and_log("Testing check_types_as_string function with Spotify Dataset...")
        test_df = self.data_dictionary.copy()

        # 1. All integer strings (create a string version of 'key')
        test_df['key_str'] = test_df['key'].astype(str)
        result = self.data_smells.check_types_as_string(test_df, 'key_str', DataType.STRING)
        assert result is False, "Test Case 1 Failed: Should warn for integer as string in key_str"
        print_and_log("Test Case 1 Passed: Integer as string detected in key_str")

        # 2. All float strings (create a string version of 'danceability')
        test_df['danceability_str'] = test_df['danceability'].astype(str)
        result = self.data_smells.check_types_as_string(test_df, 'danceability_str', DataType.STRING)
        assert result is False, "Test Case 2 Failed: Should warn for float as string in danceability_str"
        print_and_log("Test Case 2 Passed: Float as string detected in danceability_str")

        # 3. All date strings (simulate a date column as string)
        test_df['date_str'] = ['2024-06-24'] * len(test_df)
        result = self.data_smells.check_types_as_string(test_df, 'date_str', DataType.STRING)
        assert result is False, "Test Case 3 Failed: Should warn for date as string in date_str"
        print_and_log("Test Case 3 Passed: Date as string detected in date_str")

        # 4. Mixed string values (should not warn)
        test_df['mixed_str'] = ['abc', '123', '2024-06-24'] * (len(test_df) // 3) + ['abc'] * (len(test_df) % 3)
        result = self.data_smells.check_types_as_string(test_df, 'mixed_str', DataType.STRING)
        assert result is True, "Test Case 4 Failed: Should not warn for mixed string values in mixed_str"
        print_and_log("Test Case 4 Passed: Mixed string values handled correctly in mixed_str")

        # 5. Type mismatch (should raise TypeError)
        with self.assertRaises(TypeError):
            self.data_smells.check_types_as_string(test_df, 'key_str', DataType.INTEGER)
        print_and_log("Test Case 5 Passed: TypeError raised for type mismatch in key_str")

        # 6. Non-existent field (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_types_as_string(test_df, 'no_field', DataType.STRING)
        print_and_log("Test Case 6 Passed: ValueError raised for non-existent field")

        # 7. Unknown expected_type (should raise ValueError)
        with self.assertRaises(ValueError):
            self.data_smells.check_types_as_string(test_df, 'key_str', 'UnknownType')
        print_and_log("Test Case 7 Passed: ValueError raised for unknown expected_type")

        print_and_log("\nFinished testing check_types_as_string function with Spotify Dataset")
        print_and_log("")

    def execute_check_special_character_spacing_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_special_character_spacing function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_special_character_spacing Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check track_name field (likely has various formatting issues)
        print_and_log("\nTest 1: Check track_name field")
        result = self.data_smells.check_special_character_spacing(test_df, 'track_name')
        # This will likely detect issues due to special characters, parentheses, etc.
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for track_name due to special characters")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: Check track_artist field (likely has various formatting issues)
        print_and_log("\nTest 2: Check track_artist field")
        result = self.data_smells.check_special_character_spacing(test_df, 'track_artist')
        # This will likely detect issues due to special characters, apostrophes, etc.
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for track_artist due to special characters")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Check a numeric field (should have no smell)
        print_and_log("\nTest 3: Check numeric field (danceability)")
        result = self.data_smells.check_special_character_spacing(test_df, 'danceability')
        self.assertTrue(result, "Test Case 3 Failed: Expected no smell for numeric field")
        print_and_log("Test Case 3 Passed: Expected no smell, got no smell")

        # Test 4: Create a clean string column (no smell)
        print_and_log("\nTest 4: Check clean string column")
        clean_df = pd.DataFrame({
            'clean_strings': ['clean text', 'simple words', 'normal string', 'good data', 'perfect text']
        })
        result = self.data_smells.check_special_character_spacing(clean_df, 'clean_strings')
        self.assertTrue(result, "Test Case 4 Failed: Expected no smell for clean strings")
        print_and_log("Test Case 4 Passed: Expected no smell, got no smell")

        # Test 5: Create a column with uppercase issues (smell)
        print_and_log("\nTest 5: Check uppercase issues")
        uppercase_df = pd.DataFrame({
            'uppercase_text': ['Hello World', 'TEST CASE', 'Simple Text', 'UPPERCASE', 'MixedCase']
        })
        result = self.data_smells.check_special_character_spacing(uppercase_df, 'uppercase_text')
        self.assertFalse(result, "Test Case 5 Failed: Expected smell for uppercase text")
        print_and_log("Test Case 5 Passed: Expected smell, got smell")

        # Test 6: Create a column with accented characters (smell)
        print_and_log("\nTest 6: Check accented characters")
        accent_df = pd.DataFrame({
            'accented_text': ['café', 'niño', 'résumé', 'piñata', 'naïve']
        })
        result = self.data_smells.check_special_character_spacing(accent_df, 'accented_text')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for accented text")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Create a column with special characters (smell)
        print_and_log("\nTest 7: Check special characters")
        special_df = pd.DataFrame({
            'special_chars': ['hello@world', 'test#case', 'data&analytics', 'file.txt', 'user:password']
        })
        result = self.data_smells.check_special_character_spacing(special_df, 'special_chars')
        self.assertFalse(result, "Test Case 7 Failed: Expected smell for special characters")
        print_and_log("Test Case 7 Passed: Expected smell, got smell")

        # Test 8: Create a column with extra spacing (smell)
        print_and_log("\nTest 8: Check extra spacing")
        spacing_df = pd.DataFrame({
            'extra_spaces': ['hello  world', 'test   case', 'data    analytics', '  leading', 'trailing  ']
        })
        result = self.data_smells.check_special_character_spacing(spacing_df, 'extra_spaces')
        self.assertFalse(result, "Test Case 8 Failed: Expected smell for extra spacing")
        print_and_log("Test Case 8 Passed: Expected smell, got smell")

        # Test 9: Create a column with mixed formatting issues (smell)
        print_and_log("\nTest 9: Check mixed formatting issues")
        mixed_df = pd.DataFrame({
            'mixed_issues': ['Café@Home  ', 'TEST#Case   ', 'Résumé!Final  ', 'USER:Pass  ', 'File.TXT@Server']
        })
        result = self.data_smells.check_special_character_spacing(mixed_df, 'mixed_issues')
        self.assertFalse(result, "Test Case 9 Failed: Expected smell for mixed issues")
        print_and_log("Test Case 9 Passed: Expected smell, got smell")

        # Test 10: Check numbers as strings (no smell)
        print_and_log("\nTest 10: Check numbers as strings")
        numbers_df = pd.DataFrame({
            'numbers_as_strings': ['123', '456', '789', '101112', '131415']
        })
        result = self.data_smells.check_special_character_spacing(numbers_df, 'numbers_as_strings')
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for numbers as strings")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Check empty and null values (no smell)
        print_and_log("\nTest 11: Check empty and null values")
        empty_df = pd.DataFrame({
            'empty_nulls': ['', np.nan, '', np.nan, '']
        })
        result = self.data_smells.check_special_character_spacing(empty_df, 'empty_nulls')
        self.assertTrue(result, "Test Case 11 Failed: Expected no smell for empty/null values")
        print_and_log("Test Case 11 Passed: Expected no smell, got no smell")

        # Test 12: Check single characters with issues (smell)
        print_and_log("\nTest 12: Check single character issues")
        single_char_df = pd.DataFrame({
            'single_chars': ['A', '@', 'É', ' ', '#']
        })
        result = self.data_smells.check_special_character_spacing(single_char_df, 'single_chars')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for single character issues")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Check mixed clean and dirty data (smell)
        print_and_log("\nTest 13: Check mixed clean and dirty data")
        mixed_clean_dirty_df = pd.DataFrame({
            'mixed_data': ['clean text', 'Dirty@Text  ', 'normal', 'UPPERCASE', 'café']
        })
        result = self.data_smells.check_special_character_spacing(mixed_clean_dirty_df, 'mixed_data')
        self.assertFalse(result, "Test Case 13 Failed: Expected smell for mixed clean/dirty data")
        print_and_log("Test Case 13 Passed: Expected smell, got smell")

        # Test 14: Check all string columns in dataset (smell expected)
        print_and_log("\nTest 14: Check all string columns in dataset")
        result = self.data_smells.check_special_character_spacing(test_df)
        self.assertFalse(result, "Test Case 14 Failed: Expected smell when checking all string columns")
        print_and_log("Test Case 14 Passed: Expected smell when checking all columns, got smell")

        # Test 15: Check playlist_name field (likely has formatting issues)
        print_and_log("\nTest 15: Check playlist_name field")
        result = self.data_smells.check_special_character_spacing(test_df, 'playlist_name')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for playlist_name due to formatting")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

    def execute_check_suspect_distribution_ExternalDatasetTests(self):
        """
        Execute the external dataset tests for check_suspect_distribution function
        Tests various scenarios with the Spotify dataset
        """
        print_and_log("Testing check_suspect_distribution Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()

        # Test 1: Check danceability field (should be 0.0-1.0, no smell expected)
        print_and_log("\nTest 1: Check danceability field (0.0-1.0 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0.0, 1.0, 'danceability')
        self.assertTrue(result, "Test Case 1 Failed: Expected no smell for danceability in range 0.0-1.0")
        print_and_log("Test Case 1 Passed: Expected no smell, got no smell")

        # Test 2: Check danceability with restrictive range (smell expected)
        print_and_log("\nTest 2: Check danceability with restrictive range (0.3-0.7)")
        result = self.data_smells.check_suspect_distribution(test_df, 0.3, 0.7, 'danceability')
        self.assertFalse(result, "Test Case 2 Failed: Expected smell for danceability with restrictive range")
        print_and_log("Test Case 2 Passed: Expected smell, got smell")

        # Test 3: Check track_popularity (0-100 range, no smell expected)
        print_and_log("\nTest 3: Check track_popularity field (0-100 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0, 100, 'track_popularity')
        self.assertTrue(result, "Test Case 3 Failed: Expected no smell for track_popularity in range 0-100")
        print_and_log("Test Case 3 Passed: Expected no smell, got no smell")

        # Test 4: Check track_popularity with restrictive range (smell expected)
        print_and_log("\nTest 4: Check track_popularity with restrictive range (20-80)")
        result = self.data_smells.check_suspect_distribution(test_df, 20, 80, 'track_popularity')
        self.assertFalse(result, "Test Case 4 Failed: Expected smell for track_popularity with restrictive range")
        print_and_log("Test Case 4 Passed: Expected smell, got smell")

        # Test 5: Check duration_ms with reasonable range (no smell expected)
        print_and_log("\nTest 5: Check duration_ms field (1000-1000000 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 1000, 1000000, 'duration_ms')
        self.assertTrue(result, "Test Case 5 Failed: Expected no smell for duration_ms in reasonable range")
        print_and_log("Test Case 5 Passed: Expected no smell, got no smell")

        # Test 6: Check duration_ms with restrictive range (smell expected)
        print_and_log("\nTest 6: Check duration_ms with restrictive range (150000-250000)")
        result = self.data_smells.check_suspect_distribution(test_df, 150000, 250000, 'duration_ms')
        self.assertFalse(result, "Test Case 6 Failed: Expected smell for duration_ms with restrictive range")
        print_and_log("Test Case 6 Passed: Expected smell, got smell")

        # Test 7: Check tempo field with reasonable range (no smell expected)
        print_and_log("\nTest 7: Check tempo field (0-250 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0, 250, 'tempo')
        self.assertTrue(result, "Test Case 7 Failed: Expected no smell for tempo in reasonable range")
        print_and_log("Test Case 7 Passed: Expected no smell, got no smell")

        # Test 8: Check key field (0-11 range, no smell expected)
        print_and_log("\nTest 8: Check key field (0-11 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0, 11, 'key')
        self.assertTrue(result, "Test Case 8 Failed: Expected no smell for key in range 0-11")
        print_and_log("Test Case 8 Passed: Expected no smell, got no smell")

        # Test 9: Check mode field (0-1 range, no smell expected)
        print_and_log("\nTest 9: Check mode field (0-1 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0, 1, 'mode')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for mode in range 0-1")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: Check energy field (0.0-1.0 range, no smell expected)
        print_and_log("\nTest 10: Check energy field (0.0-1.0 range)")
        result = self.data_smells.check_suspect_distribution(test_df, 0.0, 1.0, 'energy')
        self.assertTrue(result, "Test Case 10 Failed: Expected no smell for energy in range 0.0-1.0")
        print_and_log("Test Case 10 Passed: Expected no smell, got no smell")

        # Test 11: Create custom data with out-of-range values (smell expected)
        print_and_log("\nTest 11: Check custom data with out-of-range values")
        custom_df = pd.DataFrame({
            'test_values': [0.5, 0.8, 1.2, 0.3, 0.9]  # 1.2 is out of range for 0.0-1.0
        })
        result = self.data_smells.check_suspect_distribution(custom_df, 0.0, 1.0, 'test_values')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for out-of-range values")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Create custom data with negative values (smell expected)
        print_and_log("\nTest 12: Check custom data with negative values")
        negative_df = pd.DataFrame({
            'negative_values': [-0.1, 0.5, 0.8, 0.3, 0.9]  # -0.1 is out of range for 0.0-1.0
        })
        result = self.data_smells.check_suspect_distribution(negative_df, 0.0, 1.0, 'negative_values')
        self.assertFalse(result, "Test Case 12 Failed: Expected smell for negative values")
        print_and_log("Test Case 12 Passed: Expected smell, got smell")

        # Test 13: Check string field (no smell - non-numeric)
        print_and_log("\nTest 13: Check string field (track_name)")
        result = self.data_smells.check_suspect_distribution(test_df, 0.0, 1.0, 'track_name')
        self.assertTrue(result, "Test Case 13 Failed: Expected no smell for string field")
        print_and_log("Test Case 13 Passed: Expected no smell for string field, got no smell")

        # Test 14: Check all numeric columns with broad range (some smells expected)
        print_and_log("\nTest 14: Check all numeric columns with restrictive range")
        result = self.data_smells.check_suspect_distribution(test_df, 0.0, 1.0)  # Very restrictive range
        self.assertFalse(result, "Test Case 14 Failed: Expected smell when checking all columns with restrictive range")
        print_and_log("Test Case 14 Passed: Expected smell when checking all columns, got smell")

        # Test 15: Check loudness field with reasonable range (no smell expected)
        print_and_log("\nTest 15: Check loudness field (-50 to 5 range)")
        result = self.data_smells.check_suspect_distribution(test_df, -50, 5, 'loudness')
        self.assertTrue(result, "Test Case 15 Failed: Expected no smell for loudness in reasonable range")
        print_and_log("Test Case 15 Passed: Expected no smell, got no smell")

    def execute_check_ambiguous_datetime_format_ExternalDatasetTests(self):
        """
        Execute external dataset tests for check_ambiguous_datetime_format function.
        Tests various scenarios with the Spotify dataset by adding datetime columns with different formats.
        """
        print_and_log("Testing check_ambiguous_datetime_format Function with Spotify Dataset")
        print_and_log("")

        # Create a copy of the dataset for modifications
        test_df = self.data_dictionary.copy()
        n_rows = len(test_df)

        # Test 1: Column with 12-hour format values (smell detected)
        print_and_log("\nTest 1: Check column with 12-hour AM/PM format")
        test_df['datetime_12h'] = ['01/15/2023 02:30 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_12h')
        self.assertFalse(result, "Test Case 1 Failed: Expected smell for 12-hour format")
        print_and_log("Test Case 1 Passed: Expected smell, got smell")

        # Test 2: Column with 24-hour format values (no smell)
        print_and_log("\nTest 2: Check column with 24-hour format")
        test_df['datetime_24h'] = ['2023-01-15 14:30:00'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_24h')
        self.assertTrue(result, "Test Case 2 Failed: Expected no smell for 24-hour format")
        print_and_log("Test Case 2 Passed: Expected no smell, got no smell")

        # Test 3: Column with a.m./p.m. format (smell detected)
        print_and_log("\nTest 3: Check column with a.m./p.m. format")
        test_df['datetime_am_pm'] = ['01/15/2023 02:30 a.m.'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_am_pm')
        self.assertFalse(result, "Test Case 3 Failed: Expected smell for a.m./p.m. format")
        print_and_log("Test Case 3 Passed: Expected smell, got smell")

        # Test 4: Column with mixed 12-hour patterns (smell detected)
        print_and_log("\nTest 4: Check column with mixed 12-hour patterns")
        # Create a mixed pattern by cycling through different values
        patterns = ['2:30 PM', '9:15am', '11:45 p.m.', '1:00 AM']
        test_df['datetime_mixed'] = [patterns[i % len(patterns)] for i in range(n_rows)]
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_mixed')
        self.assertFalse(result, "Test Case 4 Failed: Expected smell for mixed 12-hour patterns")
        print_and_log("Test Case 4 Passed: Expected smell, got smell")

        # Test 5: Column with date-only values (no smell)
        print_and_log("\nTest 5: Check column with date-only values")
        test_df['date_only'] = ['2023-01-15'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'date_only')
        self.assertTrue(result, "Test Case 5 Failed: Expected no smell for date-only format")
        print_and_log("Test Case 5 Passed: Expected no smell, got no smell")

        # Test 6: Column with time patterns that could be 12-hour but without AM/PM (no smell)
        print_and_log("\nTest 6: Check column with ambiguous time patterns without AM/PM")
        test_df['time_no_ampm'] = ['02:30'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'time_no_ampm')
        self.assertTrue(result, "Test Case 6 Failed: Expected no smell for time without AM/PM indicators")
        print_and_log("Test Case 6 Passed: Expected no smell, got no smell")

        # Test 7: Column with mixed 24-hour and 12-hour formats (smell detected)
        print_and_log("\nTest 7: Check column with mixed 24-hour and 12-hour formats")
        # Create a mix where some values have AM/PM (will trigger smell)
        mixed_patterns = ['2023-01-15 14:30:00', '01/15/2023 02:30 PM', '2023-01-15 23:45:00']
        test_df['mixed_formats'] = [mixed_patterns[i % len(mixed_patterns)] for i in range(n_rows)]
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'mixed_formats')
        self.assertFalse(result, "Test Case 7 Failed: Expected smell for mixed formats")
        print_and_log("Test Case 7 Passed: Expected smell, got smell")

        # Test 8: Non-existent column (error)
        print_and_log("\nTest 8: Check with non-existent column")
        with self.assertRaises(ValueError):
            self.data_smells.check_ambiguous_datetime_format(test_df, 'non_existent_column')
        print_and_log("Test Case 8 Passed: Expected ValueError for non-existent column")

        # Test 9: Column with non-datetime string values (no smell)
        print_and_log("\nTest 9: Check column with non-datetime string values")
        test_df['non_datetime'] = ['hello world'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'non_datetime')
        self.assertTrue(result, "Test Case 9 Failed: Expected no smell for non-datetime values")
        print_and_log("Test Case 9 Passed: Expected no smell, got no smell")

        # Test 10: Column with 12-hour format with seconds (smell detected)
        print_and_log("\nTest 10: Check column with 12-hour format including seconds")
        test_df['datetime_12h_seconds'] = ['01/15/2023 02:30:45 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_12h_seconds')
        self.assertFalse(result, "Test Case 10 Failed: Expected smell for 12-hour format with seconds")
        print_and_log("Test Case 10 Passed: Expected smell, got smell")

        # Test 11: Column with uppercase AM/PM (smell detected)
        print_and_log("\nTest 11: Check column with uppercase AM/PM")
        test_df['datetime_upper'] = ['01/15/2023 02:30 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_upper')
        self.assertFalse(result, "Test Case 11 Failed: Expected smell for uppercase AM/PM")
        print_and_log("Test Case 11 Passed: Expected smell, got smell")

        # Test 12: Column with numeric values (using existing numeric column - no smell)
        print_and_log("\nTest 12: Check column with numeric values")
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'danceability')
        self.assertTrue(result, "Test Case 12 Failed: Expected no smell for numeric values")
        print_and_log("Test Case 12 Passed: Expected no smell, got no smell")

        # Test 13: Column with NaN and valid 12-hour format (smell detected)
        print_and_log("\nTest 13: Check column with NaN and 12-hour format")
        # Create a pattern with mostly NaN but some 12-hour format
        test_values = [np.nan] * n_rows
        test_values[0] = '01/15/2023 02:30 PM'
        test_values[1] = '01/15/2023 09:15 AM'
        test_df['datetime_with_nan'] = test_values
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_with_nan')
        self.assertFalse(result, "Test Case 13 Failed: Expected smell for 12-hour format with NaN")
        print_and_log("Test Case 13 Passed: Expected smell, got smell")

        # Test 14: Column with empty strings and 24-hour format (no smell)
        print_and_log("\nTest 14: Check column with empty strings and 24-hour format")
        test_values_empty = [''] * n_rows
        test_values_empty[0] = '2023-01-15 14:30:00'
        test_values_empty[1] = '2023-01-15 09:15:00'
        test_df['datetime_with_empty'] = test_values_empty
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'datetime_with_empty')
        self.assertTrue(result, "Test Case 14 Failed: Expected no smell for 24-hour format with empty strings")
        print_and_log("Test Case 14 Passed: Expected no smell, got no smell")

        # Test 15: Column with only time in 12-hour format (smell detected)
        print_and_log("\nTest 15: Check column with only time in 12-hour format")
        test_df['time_12h_only'] = ['02:30 PM'] * n_rows
        result = self.data_smells.check_ambiguous_datetime_format(test_df, 'time_12h_only')
        self.assertFalse(result, "Test Case 15 Failed: Expected smell for 12-hour time only")
        print_and_log("Test Case 15 Passed: Expected smell, got smell")

        print_and_log("\nFinished testing check_ambiguous_datetime_format function with Spotify Dataset")
        print_and_log("--------------------------------------------------------------------------")
