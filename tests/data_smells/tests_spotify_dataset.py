# Importing libraries
import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm
import functions.data_smells as data_smells

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
            self.execute_check_integer_as_floating_point_ExternalDatasetTests
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

        print_and_log("\nFinished testing check_integer_as_floating_point function with Spotify Dataset")
        print_and_log("-----------------------------------------------------------")
