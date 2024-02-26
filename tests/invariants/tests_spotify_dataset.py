import os
import unittest

import pandas as pd

from functions.contract_invariants import ContractsInvariants
from helpers.logger import print_and_log

#TODO: Implement the invariants tests with external dataset
class InvariantsExternalDatasetTests(unittest.TestCase):
    """
        Class to test the invariants with external dataset test cases

        Attributes:
        pre_post (ContractsPrePost): instance of the class ContractsPrePost
        dataDictionary (pd.DataFrame): dataframe with the external dataset. It must be loaded in the __init__ method


        Methods:

    """
    def __init__(self):
        """
        Constructor of the class

        Atributes:
        invariants (ContractsInvariants): instance of the class ContractsInvariants
        data_dictionary (pd.DataFrame): dataframe with the external dataset. It must be loaded in the __init__ method
        """
        self.invariants = ContractsInvariants()

        # Obtiene la ruta del directorio actual del script
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        # Construye la ruta al archivo CSV
        ruta_csv = os.path.join(directorio_actual, '../../test_datasets/spotify_songs/spotify_songs.csv')
        # Crea el dataframe a partir del archivo CSV
        self.data_dictionary = pd.read_csv(ruta_csv)


    def executeAll_ExternalDatasetTests(self):
        """
        Execute all the invariants with external dataset tests
        """
        test_methods = [
            self.execute_CheckInv_FixValue_FixValue_ExternalDatasetTests,
            self.execute_checkInv_FixValue_DerivedValue_ExternalDatasetTests,
            self.execute_checkInv_FixValue_NumOp_ExternalDatasetTests,
            self.execute_checkInv_Interval_FixValue_ExternalDatasetTests,
            self.execute_checkInv_Interval_DerivedValue_ExternalDatasetTests,
            self.execute_checkInv_Interval_NumOp_ExternalDatasetTests,
            self.execute_checkInv_SpecialValue_FixValue_ExternalDatasetTests,
            self.execute_checkInv_SpecialValue_DerivedValue_ExternalDatasetTests,
            self.execute_checkInv_SpecialValue_NumOp_ExternalDatasetTests
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("----- STARTING INVARIANT DATASET TEST CASES ------")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for test_method in test_methods:
            test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ DATASET INVARIANT TEST CASES FINISHED -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")


    # TODO: Implement the invariant tests with external dataset
    def execute_CheckInv_FixValue_FixValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_FixValue
        """
        print_and_log("Testing checkInv_FixValue_FixValue Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_checkInv_FixValue_DerivedValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_DerivedValue
        """
        print_and_log("Testing checkInv_FixValue_DerivedValue Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_checkInv_FixValue_NumOp_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_NumOp
        """
        print_and_log("Testing checkInv_FixValue_NumOp Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_checkInv_Interval_FixValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_FixValue
        """
        print_and_log("Testing checkInv_Interval_FixValue Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_checkInv_Interval_DerivedValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_DerivedValue
        """
        print_and_log("Testing checkInv_Interval_DerivedValue Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_checkInv_Interval_NumOp_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_NumOp
        """
        print_and_log("Testing checkInv_Interval_NumOp Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_checkInv_SpecialValue_FixValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_FixValue
        """
        print_and_log("Testing checkInv_SpecialValue_FixValue Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_checkInv_SpecialValue_DerivedValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_DerivedValue
        """
        print_and_log("Testing checkInv_SpecialValue_DerivedValue Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_checkInv_SpecialValue_NumOp_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_NumOp
        """
        print_and_log("Testing checkInv_SpecialValue_NumOp Invariant Function")
        print_and_log("")

        print_and_log("Test cases with external dataset added:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")