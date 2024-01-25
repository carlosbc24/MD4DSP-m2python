import os
import unittest

import pandas as pd

from functions.contract_pre_post import ContractsPrePost
from helpers.enumerations import Belong, Operator, Closure
from helpers.logger import print_and_log


class ContractTestWithDatasets(unittest.TestCase):
    """
    Class to test the contracts with simple test cases

    Attributes:
    pre_post (ContractsPrePost): instance of the class ContractsPrePost

    Methods:
    execute_CheckFieldRange_Tests: execute the simple tests of the function checkFieldRange
    execute_CheckFixValueRangeString_Tests: execute the simple tests of the function checkFixValueRangeString
    execute_CheckFixValueRangeFloat_Tests: execute the simple tests of the function checkFixValueRangeFloat
    execute_CheckFixValueRangeDateTime_Tests: execute the simple tests of the function checkFixValueRangeDateTime
    """
    def __init__(self):
        """
        Constructor of the class

        Attributes:
        pre_post (ContractsPrePost): instance of the class ContractsPrePost
        """
        self.pre_post = ContractsPrePost()


    def execute_CheckFieldRange_Tests(self):
        """
        Execute the datasets_tests of the checkFieldRange function
        """
        print_and_log("Testing CheckFieldRange Function")
        print_and_log("")

        print_and_log("Casos Básicos solicitados en la especificación del contrato:")

        # Obtiene la ruta del directorio actual del script
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        # Construye la ruta al archivo CSV
        ruta_csv = os.path.join(directorio_actual, '../test_datasets/spotify_songs/spotify_songs.csv')
        #Crea el dataframe a partir del archivo CSV
        data_dictionary = pd.read_csv(ruta_csv)

        # Case 1 of checkFieldRange
        # Check that fields 'c1' and 'c2' belong to the data dictionary. It must return True
        fields = ['track_id', 'loudness', 'playlist_id']
        belong = 0
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Case 2 of checkFieldRange
        # Check that fields 'c2' and 'c3' belong to the data dictionary. It must return False as 'c3' does not belong
        fields = ['track_id', 'loudness', 'c3']
        belong = 0
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Case 3 of checkFieldRange
        # Check that fields 'c2' and 'c3' don't belong to the data dictionary.It must return True as 'c2' doesn't belong
        fields = ['c2', 'track_id', 'loudness']
        belong = 1
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Case 4 of checkFieldRange
        # Check that fields 'c1' and 'c2' don't belong to the data dictionary. It must return False as both belong
        fields = ['track_id', 'loudness', 'valence']
        belong = 1
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")





























