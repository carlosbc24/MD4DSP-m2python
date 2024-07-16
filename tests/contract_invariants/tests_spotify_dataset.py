import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

import functions.contract_invariants as invariants
import functions.data_transformations as data_transformations
from helpers.enumerations import Closure, DataType, SpecialType, Belong
from helpers.enumerations import DerivedType, Operation
from helpers.logger import print_and_log


class InvariantsExternalDatasetTests(unittest.TestCase):
    """
        Class to test the invariant with external dataset test cases

        Attributes:
        unittest.TestCase: class that inherits from unittest.TestCase

        Methods:
        executeAll_ExternalDatasetTests: execute all the invariant with external dataset tests
        execute_checkInv_FixValue_FixValue: execute the invariant test with external dataset for
        the function checkInv_FixValue_FixValue execute_SmallBatchTests_checkInv_FixValue_FixValue:
        execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_FixValue
        execute_WholeDatasetTests_checkInv_FixValue_FixValue: execute the invariant test using the
        whole dataset for the function checkInv_FixValue_FixValue
        execute_checkInv_FixValue_DerivedValue: execute the invariant test with external dataset
        for the function checkInv_FixValue_DerivedValue
        execute_SmallBatchTests_checkInv_FixValue_DerivedValue: execute the invariant test using a
        small batch of the dataset for the function checkInv_FixValue_DerivedValue
        execute_WholeDatasetTests_checkInv_FixValue_DerivedValue: execute the invariant test using
        the whole dataset for the function checkInv_FixValue_DerivedValue
        execute_checkInv_FixValue_NumOp: execute the invariant test with external dataset for
        the function checkInv_FixValue_NumOp execute_SmallBatchTests_checkInv_FixValue_NumOp: execute
        the invariant test using a small batch of the dataset for the function checkInv_FixValue_NumOp
        execute_WholeDatasetTests_checkInv_FixValue_NumOp: execute the invariant test using the whole
        dataset for the function checkInv_FixValue_NumOp execute_checkInv_Interval_FixValue:
        execute the invariant test with external dataset for the function checkInv_Interval_FixValue
        execute_SmallBatchTests_checkInv_Interval_FixValue: execute the invariant test using a small
        batch of the dataset for the function checkInv_Interval_FixValue
        execute_WholeDatasetTests_checkInv_Interval_FixValue: execute the invariant test using the
        whole dataset for the function checkInv_Interval_FixValue
        execute_checkInv_Interval_DerivedValue: execute the invariant test with external dataset
        for the function checkInv_Interval_DerivedValue
        execute_SmallBatchTests_checkInv_Interval_DerivedValue: execute the invariant test using a
        small batch of the dataset for the function checkInv_Interval_DerivedValue
        execute_WholeDatasetTests_checkInv_Interval_DerivedValue: execute the invariant test using
        the whole dataset for the function checkInv_Interval_DerivedValue
        execute_checkInv_Interval_NumOp: execute the invariant test with external dataset for
        the function checkInv_Interval_NumOp execute_SmallBatchTests_checkInv_Interval_NumOp: execute
        the invariant test using a small batch of the dataset for the function checkInv_Interval_NumOp
        execute_WholeDatasetTests_checkInv_Interval_NumOp: execute the invariant test using the whole
        dataset for the function checkInv_Interval_NumOp execute_checkInv_SpecialValue_FixValue:
        execute the invariant test with external dataset for the function checkInv_SpecialValue_FixValue
        execute_SmallBatchTests_checkInv_SpecialValue_FixValue: execute the invariant test using a
        small batch of the dataset for the function checkInv_SpecialValue_FixValue
        execute_WholeDatasetTests_checkInv_SpecialValue_FixValue: execute the invariant test using
        the whole dataset for the function checkInv_SpecialValue_FixValue
        execute_checkInv_SpecialValue_DerivedValue: execute the invariant test with external
        dataset for the function checkInv_SpecialValue_DerivedValue
        execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue: execute the invariant test using
        a small batch of the dataset for the function checkInv_SpecialValue_DerivedValue
        execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue: execute the invariant test
        using the whole dataset for the function checkInv_SpecialValue_DerivedValue
        execute_checkInv_SpecialValue_NumOp: execute the invariant test with external dataset
        for the function checkInv_SpecialValue_NumOp
        execute_SmallBatchTests_checkInv_SpecialValue_NumOp: execute the invariant test using a small
        batch of the dataset for the function checkInv_SpecialValue_NumOp
        execute_WholeDatasetTests_checkInv_SpecialValue_NumOp: execute the invariant test using the
        whole dataset for the function checkInv_SpecialValue_NumOp
    """

    def __init__(self):
        """
        Constructor of the class
        """
        super().__init__()
        self.invariants = invariants
        self.data_transformations = data_transformations

        # Get the current directory
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the CSV file
        ruta_csv = os.path.join(directorio_actual, '../../test_datasets/spotify_songs/spotify_songs.csv')
        # Create the dataframe with the external dataset
        self.data_dictionary = pd.read_csv(ruta_csv)

        # Select a small batch of the dataset (first 10 rows)
        self.small_batch_dataset = self.data_dictionary.head(10)
        # Select the rest of the dataset (from row 11 to the end)
        self.rest_of_dataset = self.data_dictionary.iloc[10:].reset_index(drop=True)

    def executeAll_ExternalDatasetTests(self):
        """
        Execute all the invariants with external dataset tests
        """
        test_methods = [
            self.execute_checkInv_FixValue_FixValue,
            self.execute_checkInv_FixValue_DerivedValue,
            self.execute_checkInv_FixValue_NumOp,
            self.execute_checkInv_Interval_FixValue,
            self.execute_checkInv_Interval_DerivedValue,
            self.execute_checkInv_Interval_NumOp,
            self.execute_checkInv_SpecialValue_FixValue,
            self.execute_checkInv_SpecialValue_DerivedValue,
            self.execute_checkInv_SpecialValue_NumOp,
            self.execute_checkInv_MissingValue_MissingValue
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("----- STARTING INVARIANT DATASET TEST CASES ------")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for test_method in tqdm(test_methods, desc="Running Invariant External Dataset Tests", unit="test"):
            test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ INVARIANT DATASET TEST CASES FINISHED -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")


    def execute_checkInv_FixValue_FixValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_FixValue
        """
        print_and_log("Testing checkInv_FixValue_FixValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_FixValue_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_FixValue_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_FixValue_FixValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_FixValue
        """

        # Caso 1
        # Ejecutar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            field_in=field_in, field_out=field_out,
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                         data_type_input_list=None,
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output,
                                                                         data_type_output_list=None,
                                                                         field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar el valor fijo string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = [3.0]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar el valor fijo de tipo FLOAT 2.33e-5 por el valor fijo de tipo STRING "Near 0"
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = [2.33e-5]
        fix_value_output = ["Near 0"]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            field_in=field_in, field_out=field_out,
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                         data_type_input_list=None,
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output,
                                                                         data_type_output_list=None, field_in=field_in,
                                                                         field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, and got False")

        # Caso 7
        # Ejecutar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, and got False")

        # Caso 8
        # Ejecutar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, and got False")

        # Caso 9
        # Definir el valor fijo y la condición para el cambio
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list)

        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         data_type_input_list=None, input_values_list=input_values_list,
                                                                         data_type_output_list=None, belong_op_out=Belong(0),
                                                                         output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Definir el valor fijo y la condición para el cambio
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=[3.1, 14])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                          data_type_input_list=None, input_values_list=input_values_list,
                                                                          data_type_output_list=None, belong_op_out=Belong(1),
                                                                          output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=[3.1, 14])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        output_values_list = [3.0]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                                input_values_list=input_values_list,
                                                                                output_values_list=output_values_list)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                   data_dictionary_out=result_df,
                                                                   data_type_input_list=None,
                                                                   input_values_list=input_values_list,
                                                                   data_type_output_list=None, belong_op_out=Belong(1),
                                                                   output_values_list=output_values_list)
        print_and_log("Test Case 13 Passed: Expected ValueError, got ValueError")

        # Caso 14
        output_values_list = [3.0, 3.0]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.small_batch_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list)

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15
        output_values_list = [3.0, 6.0]
        input_values_list = ['Maroon 5', 3.0]
        result_df = self.small_batch_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list)

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")


    def execute_WholeDatasetTests_checkInv_FixValue_FixValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_FixValue
        """

        # Caso 1
        # Ejecutar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            field_in=field_in, field_out=field_out,
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         data_type_input_list=None,
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output,
                                                                         data_type_output_list=None,
                                                                         field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar el valor fijo string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['Maroon 5']
        fix_value_output = [3.0]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar el valor fijo de tipo FLOAT 2.33e-5 por el valor fijo de tipo STRING "Near 0"
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [2.33e-5]
        fix_value_output = ["Near 0"]
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(0),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = [67]
        fix_value_output = [1]
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            field_in=field_in, field_out=field_out,
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         data_type_input_list=None,
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output,
                                                                         data_type_output_list=None,
                                                                         field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, and got False")

        # Caso 7
        # Ejecutar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['All the Time - Don Diablo Remix']
        fix_value_output = ['todos los tiempo - Don Diablo Remix']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, and got False")

        # Caso 8
        # Ejecutar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = ['2019-07-05']
        fix_value_output = [True]

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=fix_value_input,
                                                                            output_values_list=fix_value_output)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=expected_df,
                                                                         data_dictionary_out=result_df,
                                                                         belong_op_in=Belong(0),
                                                                         belong_op_out=Belong(1),
                                                                         input_values_list=fix_value_input,
                                                                         output_values_list=fix_value_output)
        assert result_invariant is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, and got False")

        # Caso 9
        # Definir el valor fijo y la condición para el cambio
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         data_type_input_list=None,
                                                                         input_values_list=input_values_list,
                                                                         data_type_output_list=None, belong_op_out=Belong(0),
                                                                         output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Definir el valor fijo y la condición para el cambio
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=[3.1, 14])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list)
        result_invariant = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                         data_dictionary_out=result_df,
                                                                         data_type_input_list=None,
                                                                         input_values_list=input_values_list,
                                                                         data_type_output_list=None, belong_op_out=Belong(1),
                                                                         output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result_invariant is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12
        output_values_list = [3.0, 14]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()
        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=[3.1, 14])

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        output_values_list = [3.0]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()

        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                                input_values_list=input_values_list,
                                                                                output_values_list=output_values_list)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                   data_dictionary_out=result_df,
                                                                   data_type_input_list=None,
                                                                   input_values_list=input_values_list,
                                                                   data_type_output_list=None, belong_op_out=Belong(1),
                                                                   output_values_list=output_values_list)
        print_and_log("Test Case 13 Passed: Expected ValueError, got ValueError")

        # Caso 14
        output_values_list = [3.0, 3.0]
        input_values_list = ['Maroon 5', 'Katy Perry']
        result_df = self.rest_of_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list)

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15
        output_values_list = [3.0, 6.0]
        input_values_list = ['Maroon 5', 3.0]
        result_df = self.rest_of_dataset.copy()

        result_df = self.data_transformations.transform_fix_value_fix_value(data_dictionary=result_df,
                                                                            input_values_list=input_values_list,
                                                                            output_values_list=output_values_list)

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                               data_dictionary_out=result_df,
                                                               data_type_input_list=None,
                                                               input_values_list=input_values_list,
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               output_values_list=output_values_list)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")


    def execute_checkInv_FixValue_DerivedValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_DerivedValue
        """
        print_and_log("Testing checkInv_FixValue_DerivedValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_FixValue_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_FixValue_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_FixValue_DerivedValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_DerivedValue
        """

        # Caso 1
        # Ejecutar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(1),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(1),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(2),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Inavriante. cambair el valor fijo 'pop' de la columna 'playlist_genre' por el valor derivado 2 (next)
        # a nivel de fila sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(2), axis_param=1)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             axis_param=1)
        assert result_invariant is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Check the invariant: chenged the fixed value 0.11 of the column 'liveness'
        # by the derived value 0 (most frequent) at column level on the small batch of the test dataset.
        # On a copy of the small batch of the test dataset, change the values manually and verify if the result matches
        # the expected
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.11
        field_in = 'liveness'
        field_out = 'liveness'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Check the invariant: chenged the fixed value 'Ed Sheeran' of all the dataset by the most frequent value
        # from the small batch dataset. On a copy of the small batch of the test dataset, change the values manually and
        # verify if the result matches the expected
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'Ed Sheeran'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0))
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0))
        assert result_invariant is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7 Transformación de datos: exception after trying to gte previous value from all the dataset without
        # specifying the column or row level
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(1))
        print_and_log("Test Case 7.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0),
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(1))
        print_and_log("Test Case 7.2 Passed: the invariant function raised the expected exception")

        # Caso 8: exception after trying to get the next value from all the dataset without specifying the
        # column or row level
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            resuld_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(2))
        print_and_log("Test Case 8.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0),
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(2))
        print_and_log("Test Case 8.2 Passed: the invariant function raised the expected exception")

        # Caso 9: exception after trying to get the previous value using a column level. The field doens't exist in the
        # dataset.
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'autor_artista'
        field_out = 'autor_artista'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            resuld_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                    fix_value_input=fix_value_input,
                                                                                    derived_type_output=DerivedType(1),
                                                                                    field_in=field_in, field_out=field_out)

        print_and_log("Test Case 9.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(1),
                                                                                 field_in=field_in, field_out=field_out,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0), belong_op_out=Belong(0))
        print_and_log("Test Case 9.2 Passed: the invariant function raised the expected exception")

        # tests with NOT BELONG

        # Caso 10
        # Ejecutar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 11
        # Ejecutar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(1),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(1),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12
        # Ejecutar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(2),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")


    def execute_WholeDatasetTests_checkInv_FixValue_DerivedValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Definir el resultado esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(1),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(1),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(2),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Inavriante. cambair el valor fijo 'pop' de la columna 'playlist_genre' por el valor derivado 2 (next)
        # a nivel de fila sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(2), axis_param=0)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             axis_param=0)
        assert result_invariant is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Check the invariant: chenged the fixed value 0.11 of the column 'liveness'
        # by the derived value 0 (most frequent) at column level on the small batch of the test dataset.
        # On a copy of the small batch of the test dataset, change the values manually and verify if the result matches
        # the expected
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.11
        field_in = 'liveness'
        field_out = 'liveness'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Check the invariant: chenged the fixed value 'Ed Sheeran' of all the dataset by the most frequent value
        # from the small batch dataset. On a copy of the small batch of the test dataset, change the values manually and
        # verify if the result matches the expected
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'Ed Sheeran'
        result_df = self.data_transformations.transform_fix_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            fix_value_input=fix_value_input,
            derived_type_output=DerivedType(0))
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0))
        assert result_invariant is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Transformación de datos: exception after trying to gte previous value from all the dataset without specifying the column or
        # row level
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'playlist_genre'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(1))
        print_and_log("Test Case 7.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0),
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(1))
        print_and_log("Test Case 7.2 Passed: the invariant function raised the expected exception")

        # Caso 8: exception after trying to get the next value from all the dataset without specifying the
        # column or row level
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'playlist_genre'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            resuld_df = self.data_transformations.transform_fix_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                derived_type_output=DerivedType(2))
        print_and_log("Test Case 8.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0),
                                                                                 belong_op_out=Belong(0),
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(2))

        print_and_log("Test Case 8.2 Passed: the invariant function raised the expected exception")

        # Caso 9: exception after trying to get the previous value using a column level. The field doens't exist in the
        # dataset.
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'pop'
        field_in = 'autor_artista'
        field_out = 'autor_artista'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            resuld_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                                    fix_value_input=fix_value_input,
                                                                                    derived_type_output=DerivedType(1),
                                                                                    field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                                 fix_value_input=fix_value_input,
                                                                                 derived_type_output=DerivedType(1),
                                                                                 field_in=field_in, field_out=field_out,
                                                                                 data_dictionary_out=result_df,
                                                                                 belong_op_in=Belong(0), belong_op_out=Belong(0))

        print_and_log("Test Case 9.2 Passed: the invariant function raised the expected exception")

        # tests with NOT BELONG

        # Caso 10
        # Ejecutar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'mode'
        field_out = 'mode'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(0),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(0),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 11
        # Ejecutar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 'Katy Perry'
        field_in = 'track_artist'
        field_out = 'track_artist'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(1),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(1),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12
        # Ejecutar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = '2019-12-13'
        field_in = 'track_album_release_date'
        field_out = 'track_album_release_date'
        result_df = self.data_transformations.transform_fix_value_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                                fix_value_input=fix_value_input,
                                                                                derived_type_output=DerivedType(2),
                                                                                field_in=field_in, field_out=field_out)
        result_invariant = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=expected_df,
                                                                             data_dictionary_out=result_df,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                             fix_value_input=fix_value_input,
                                                                             derived_type_output=DerivedType(2),
                                                                             field_in=field_in, field_out=field_out)
        assert result_invariant is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")


    def execute_checkInv_FixValue_NumOp(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_NumOp
        """
        print_and_log("Testing checkInv_FixValue_NumOp Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_FixValue_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_FixValue_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_FixValue_NumOp(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_NumOp
        """
        # Caso 1
        # Ejecutar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(0), field_in=field_in,
                                                                      field_out=field_out, belong_op_in=Belong(0),
                                                                      belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(1), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(2), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(3), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(3), field_in=field_in,
                                                                      field_out=field_out)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar el valor 0 en todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 1 (media) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(1), axis_param=0)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 2 (mediana) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(2), axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 3 (más cercano) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(3), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(3), axis_param=0)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 0 (interpolación) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(0), axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field_in=None, field_out=None y se lanza una excepción ValueError
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.small_batch_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(0))
        print_and_log("Test Case 9.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          num_op_output=Operation(0))
        print_and_log("Test Case 9.2 Passed: the invariant function raised the expected exception")

        # Caso 10
        # Comprobar la excepción: se pasa una columna que no existe
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.65
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                            fix_value_input=fix_value_input,
                                                                            num_op_output=Operation(0), field_in=field_in,
                                                                            field_out=field_out)
        print_and_log("Test Case 10.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          num_op_output=Operation(0), field_in=field_in,
                                                                          field_out=field_out)
        print_and_log("Test Case 10.2 Passed: the invariant function raised the expected exception")

        # Casos NOT BELONG

        # Caso 11
        # Ejecutar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(0), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12
        # Ejecutar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(1), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

        # Caso 13
        # Ejecutar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(2), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(3), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                      num_op_output=Operation(3), field_in=field_in,
                                                                      field_out=field_out)
        assert invariant_result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")


    def execute_WholeDatasetTests_checkInv_FixValue_NumOp(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_NumOp
        """
        # Caso 1
        # Ejecutar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(0), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(1), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(2), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        vaue_error_exception = ValueError
        with self.assertRaises(vaue_error_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                             fix_value_input=fix_value_input,
                                                                             num_op_output=Operation(3), field_in=field_in,
                                                                             field_out=field_out)
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          num_op_output=Operation(3), field_in=field_in,
                                                                          field_out=field_out)
        print_and_log("Test Case 4 Passed: the function raised the expected exception")

        # Caso 5
        # Ejecutar la invariante: cambiar el valor 0 en todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 1 (media) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(1), axis_param=0)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 2 (mediana) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(2), axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 3 (más cercano) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        value_error_exception = ValueError
        with self.assertRaises(value_error_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                             fix_value_input=fix_value_input,
                                                                             num_op_output=Operation(3), axis_param=0)
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          num_op_output=Operation(3), axis_param=0)
        print_and_log("Test Case 7 Passed: the function raised the expected exception")

        # Caso 8
        # Ejecutar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 0 (interpolación) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), axis_param=0)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      num_op_output=Operation(0), axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field_in=None, field_out=None y se lanza una excepción ValueError
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(0))
        print_and_log("Test Case 9.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          num_op_output=Operation(0))

        print_and_log("Test Case 9.2 Passed: the invariant function raised the expected exception")

        # Caso 10
        # Comprobar la excepción: se pasa una columna que no existe
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.65
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(
                data_dictionary=self.rest_of_dataset.copy(),
                fix_value_input=fix_value_input,
                num_op_output=Operation(0), field_in=field_in, field_out=field_out)
        print_and_log("Test Case 10.1 Passed: the transformation function raised the expected exception")
        with self.assertRaises(expected_exception):
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          num_op_output=Operation(0), field_in=field_in,
                                                                          field_out=field_out)
        print_and_log("Test Case 10.2 Passed: the invariant function raised the expected exception")

        # Casos NOT BELONG

        # Caso 11
        # Ejecutar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(0), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(0), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12
        # Ejecutar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 0.725
        field_in = 'valence'
        field_out = 'valence'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(1), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(1), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

        # Caso 13
        # Ejecutar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 8
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                         fix_value_input=fix_value_input,
                                                                         num_op_output=Operation(2), field_in=field_in,
                                                                         field_out=field_out)
        invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                      data_dictionary_out=result_df,
                                                                      fix_value_input=fix_value_input,
                                                                      num_op_output=Operation(2), field_in=field_in,
                                                                      field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fix_value_input = 99.972
        field_in = 'tempo'
        field_out = 'tempo'
        value_error_exception = ValueError
        with self.assertRaises(value_error_exception):
            result_df = self.data_transformations.transform_fix_value_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                             fix_value_input=fix_value_input,
                                                                             num_op_output=Operation(3), field_in=field_in,
                                                                             field_out=field_out)
            invariant_result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=expected_df,
                                                                          data_dictionary_out=result_df,
                                                                          fix_value_input=fix_value_input,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                          num_op_output=Operation(3), field_in=field_in,
                                                                          field_out=field_out)
        print_and_log("Test Case 14 Passed: the function raised the expected exception")


    def execute_checkInv_Interval_FixValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_FixValue
        """
        print_and_log("Testing checkInv_Interval_FixValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_Interval_FixValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_Interval_FixValue
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=65, right_margin=69,
                                                                        closure_type=Closure(3),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output='65<=Pop<=69', field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0), fix_value_output=2)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output=2)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        field_in = 'track_name'
        field_out = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.small_batch_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_interval_fix_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                left_margin=65, right_margin=69, closure_type=Closure(2),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        field_in = 'speechiness'
        field_out = 'speechiness'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=0.06, right_margin=0.1270,
                                                                        closure_type=Closure(1), fix_value_output=33,
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.06, right_margin=0.1270,
                                                                        closure_type=Closure(1),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output=33, field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        field_in = 'p'
        field_out = 'p'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.small_batch_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_interval_fix_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                left_margin=65, right_margin=69, closure_type=Closure(2),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=65, right_margin=69,
                                                                        closure_type=Closure(3),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        fix_value_output='65<=Pop<=69', field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, and got False")

        # Caso 7
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0), fix_value_output=2)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        fix_value_output=2)
        assert invariant_result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, and got False")


    def execute_WholeDatasetTests_checkInv_Interval_FixValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_FixValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=65, right_margin=69,
                                                                        closure_type=Closure(3),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output='65<=Pop<=69', field_in=field_in,
                                                                        field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0), fix_value_output=2)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output=2)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        field_in = 'track_name'
        field_out = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.rest_of_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_interval_fix_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                left_margin=65, right_margin=69, closure_type=Closure(2),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        field_in = 'speechiness'
        field_out = 'speechiness'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0.06, right_margin=0.1270,
                                                                        closure_type=Closure(1), fix_value_output=33,
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.06, right_margin=0.1270,
                                                                        closure_type=Closure(1),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                        fix_value_output=33, field_in=field_in,
                                                                        field_out=field_out)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        field_in = 'p'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_interval_fix_value(
                data_dictionary=self.rest_of_dataset.copy(), left_margin=65,
                right_margin=69, closure_type=Closure(2),
                fix_value_output=101, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_interval_fix_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                left_margin=65, right_margin=69, closure_type=Closure(2),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                fix_value_output=101, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        expected_df = self.rest_of_dataset.copy()
        field_in = 'track_popularity'
        field_out = 'track_popularity'

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=65,
                                                                        right_margin=69, closure_type=Closure(3),
                                                                        data_type_output=DataType(0),
                                                                        fix_value_output='65<=Pop<=69',
                                                                        field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=65, right_margin=69,
                                                                        closure_type=Closure(3),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        fix_value_output='65<=Pop<=69', field_in=field_in,
                                                                        field_out=field_out)
        assert invariant_result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, and got False")

        # Caso 7
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_fix_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0), fix_value_output=2)
        invariant_result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=expected_df,
                                                                        data_dictionary_out=result,
                                                                        left_margin=0.5, right_margin=1,
                                                                        closure_type=Closure(0),
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        fix_value_output=2)
        assert invariant_result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, and got False")


    def execute_checkInv_Interval_DerivedValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_DerivedValue
        """
        print_and_log("Testing checkInv_Interval_DerivedValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_Interval_DerivedValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), axis_param=None)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=None)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_interval_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                left_margin=0.2, right_margin=0.4,
                closure_type=Closure(0),
                derived_type_output=DerivedType(2), axis_param=None)
            invariant_result = self.invariants.check_inv_interval_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                left_margin=0.2, right_margin=0.4, closure_type=Closure(0),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                derived_type_output=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8
        expected_df = self.small_batch_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, and got False")

        # Caso 9
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, and got False")

        # Caso 10
        expected_df = self.small_batch_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)
        assert invariant_result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 13
        expected_df = self.small_batch_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.small_batch_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")


    def execute_WholeDatasetTests_checkInv_Interval_DerivedValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), axis_param=None)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(0),
                                                                            axis_param=None)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_interval_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                left_margin=0.2, right_margin=0.4,
                closure_type=Closure(0),
                derived_type_output=DerivedType(2), axis_param=None)
            invariant_result = self.invariants.check_inv_interval_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                left_margin=0.2, right_margin=0.4, closure_type=Closure(0),
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                derived_type_output=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8
        expected_df = self.rest_of_dataset.copy()
        field_in = 'liveness'
        field_out = 'liveness'

        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(0), field_in=field_in, field_out=field_out)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(0),
                                                                            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, and got False")

        # Caso 9
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=0)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, and got False")

        # Caso 10
        expected_df = self.rest_of_dataset.copy()
        result = self.data_transformations.transform_interval_derived_value(
            data_dictionary=self.rest_of_dataset.copy(), left_margin=0.2,
            right_margin=0.4, closure_type=Closure(0),
            derived_type_output=DerivedType(1), axis_param=1)
        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(1),
                                                                            axis_param=1)
        assert invariant_result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 13
        expected_df = self.rest_of_dataset.copy()

        result = self.data_transformations.transform_interval_derived_value(data_dictionary=self.rest_of_dataset.copy(),
                                                                            left_margin=0.2, right_margin=0.4, closure_type=Closure(0),
                                                                            derived_type_output=DerivedType(2), axis_param=0)

        invariant_result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=expected_df,
                                                                            data_dictionary_out=result,
                                                                            left_margin=0.2, right_margin=0.4,
                                                                            closure_type=Closure(0),
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            derived_type_output=DerivedType(2),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")


    def execute_checkInv_Interval_NumOp(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_NumOp
        """
        print_and_log("Testing checkInv_Interval_NumOp Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_Interval_NumOp(self):
        """
        Execute the invariant test using a small batch of the dataset for the function check_inv_interval_num_op
        """

        # ------------------------------------------BELONG-BELONG----------------------------------------------
        # Caso 1
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4, closure_type=Closure(1),
                                                                       num_op_output=Operation(0), axis_param=0, field_in=None, field_out=None)
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2,
                                                           right_margin=4, closure_type=Closure(1), num_op_output=Operation(0),
                                                           axis_param=0, belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=3, right_margin=4,
                                                                       closure_type=Closure(3), num_op_output=Operation(0),
                                                                       axis_param=1)
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=3, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0), axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=3, closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=None)
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=3, closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1), axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 6
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=3, closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 8
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=4, closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=4, closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3), axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        field_in = 'T'
        field_out = 'T'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                           left_margin=2, right_margin=4, closure_type=Closure(3),
                                                                           num_op_output=Operation(0), axis_param=None, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 10.1 Passed: expected ValueError, got ValueError")


        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=field_in, field_out=field_out,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 10.2 Passed: expected ValueError, got ValueError")

        # Caso 11
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(0), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        # Definir el resultado esperado
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3),
                                                                       num_op_output=Operation(1), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3), num_op_output=Operation(2),
                                                                       axis_param=None, field_in=field_in, field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3), num_op_output=Operation(3),
                                                                       axis_param=None, field_in=field_in, field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # ------------------------------------------BELONG-NOTBELONG----------------------------------------------
        # Caso 15
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4, closure_type=Closure(1),
                                                                       num_op_output=Operation(0), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(1), num_op_output=Operation(0), axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Caso 16
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=3, right_margin=4,
                                                                       closure_type=Closure(3), num_op_output=Operation(0),
                                                                       axis_param=1, field_in=None, field_out=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=3, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0), axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Caso 17
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=1, right_margin=3, closure_type=Closure(0),
                                                                       num_op_output=Operation(1), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, got True")

        # Caso 18
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=3, closure_type=Closure(2),
                                                                       num_op_output=Operation(1), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1), axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 20
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=3, closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 20 Failed: Expected False, but got True"
        print_and_log("Test Case 20 Passed: Expected False, got False")

        # Caso 21
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=50, closure_type=Closure(2),
                                                                       num_op_output=Operation(2), axis_param=1)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2), axis_param=1,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 21 Failed: Expected True, but got False"
        print_and_log("Test Case 21 Passed: Expected True, got True")

        # Caso 22
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=4, closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, got False")

        # Caso 23
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=0, right_margin=4, closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=0, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3), axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                           left_margin=2, right_margin=4,
                                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                                           axis_param=None, field_in=None, field_out=None)
        print_and_log("Test Case 24.1 Passed: expected ValueError, got ValueError")
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=None, field_out=None,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 24.2 Passed: expected ValueError, got ValueError")

        # Caso 25
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3), num_op_output=Operation(0),
                                                                       axis_param=None, field_in=field_in, field_out=field_out)

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3), num_op_output=Operation(1),
                                                                       axis_param=None, field_in=field_in, field_out=field_out)
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 26 Failed: Expected False, but got True"
        print_and_log("Test Case 26 Passed: Expected False, got False")

        # Caso 27
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4, closure_type=Closure(3),
                                                                       num_op_output=Operation(2), axis_param=None, field_in=field_in, field_out=field_out)

        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                       left_margin=2, right_margin=4, closure_type=Closure(3),
                                                                       num_op_output=Operation(3), axis_param=None, field_in=field_in, field_out=field_out)

        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.small_batch_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 28 Failed: Expected False, but got True"
        print_and_log("Test Case 28 Passed: Expected False, got False")


    def execute_WholeDatasetTests_checkInv_Interval_NumOp(self):
        """
        Execute the invariant test using the whole dataset for the function check_inv_interval_num_op
        """

        # # ------------------------------------------BELONG-BELONG----------------------------------------------
        # Caso 1
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                      left_margin=2, right_margin=4, closure_type=Closure(1),
                                                                      num_op_output=Operation(0), axis_param=0, field_in=None, field_out=None)
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2,
                                                         right_margin=4, closure_type=Closure(1), num_op_output=Operation(0),
                                                         axis_param=0, belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                      left_margin=3, right_margin=4,
                                                                      closure_type=Closure(3), num_op_output=Operation(0),
                                                                      axis_param=1)
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=3, right_margin=4,
                                                         closure_type=Closure(3), num_op_output=Operation(0), axis_param=1,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0, right_margin=3, closure_type=Closure(0),
                                                                        num_op_output=Operation(1), axis_param=None)
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=0, right_margin=3,
                                                         closure_type=Closure(0), num_op_output=Operation(1),
                                                         axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0, right_margin=3, closure_type=Closure(0),
                                                                        num_op_output=Operation(1), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=0, right_margin=3,
                                                         closure_type=Closure(0), num_op_output=Operation(1), axis_param=0,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 6
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0, right_margin=3, closure_type=Closure(2),
                                                                        num_op_output=Operation(2), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=0, right_margin=3,
                                                         closure_type=Closure(2), num_op_output=Operation(2),
                                                         axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 8
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=2, right_margin=4, closure_type=Closure(0),
                                                                        num_op_output=Operation(3), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                         closure_type=Closure(0), num_op_output=Operation(3),
                                                         axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=2, right_margin=4, closure_type=Closure(0),
                                                                        num_op_output=Operation(3), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                         closure_type=Closure(0), num_op_output=Operation(3), axis_param=0,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        field_in = 'T'
        field_out = 'T'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=2, right_margin=4, closure_type=Closure(3),
                                                                        num_op_output=Operation(0), axis_param=None, field_in=field_in, field_out=field_out)
        print_and_log("Test Case 10.1 Passed: expected ValueError, got ValueError")


        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2,
                                                             right_margin=4,
                                                             closure_type=Closure(3), num_op_output=Operation(0),
                                                             axis_param=None, field_in=field_in, field_out=field_out,
                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                             data_dictionary_out=expected)
        print_and_log("Test Case 10.2 Passed: expected ValueError, got ValueError")

        # Caso 11
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                      left_margin=2, right_margin=4,
                                                                      closure_type=Closure(3),
                                                                      num_op_output=Operation(0), axis_param=None,
                                                                      field_in=field_in, field_out=field_out)

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                         closure_type=Closure(3), num_op_output=Operation(0),
                                                         axis_param=None, field_in=field_in, field_out=field_out, belong_op_in=Belong(0),
                                                         belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        # Definir el resultado esperado
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=2, right_margin=4,
                                                                        closure_type=Closure(3),
                                                                        num_op_output=Operation(1), axis_param=None,
                                                                        field_in=field_in, field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                         closure_type=Closure(3), num_op_output=Operation(1),
                                                         axis_param=None, field_in=field_in, field_out=field_out,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                      left_margin=2, right_margin=4,
                                                                      closure_type=Closure(3), num_op_output=Operation(2),
                                                                      axis_param=None, field_in=field_in, field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                         closure_type=Closure(3), num_op_output=Operation(2),
                                                         axis_param=None, field_in=field_in, field_out=field_out,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                      left_margin=2, right_margin=4,
                                                                      closure_type=Closure(3), num_op_output=Operation(3),
                                                                      axis_param=None, field_in=field_in, field_out=field_out)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                         closure_type=Closure(3), num_op_output=Operation(3),
                                                         axis_param=None, field_in=field_in, field_out=field_out,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                         data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # ------------------------------------------BELONG-NOTBELONG----------------------------------------------
        # Caso 15
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=2, right_margin=4, closure_type=Closure(1),
                                                                        num_op_output=Operation(0), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                         closure_type=Closure(1), num_op_output=Operation(0), axis_param=0,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Caso 16
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                      left_margin=3, right_margin=4,
                                                                      closure_type=Closure(3), num_op_output=Operation(0),
                                                                      axis_param=1, field_in=None, field_out=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=3, right_margin=4,
                                                         closure_type=Closure(3), num_op_output=Operation(0), axis_param=1,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Caso 17
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                      left_margin=1, right_margin=3, closure_type=Closure(0),
                                                                      num_op_output=Operation(1), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=0, right_margin=3,
                                                         closure_type=Closure(0), num_op_output=Operation(1),
                                                         axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, got True")

        # Caso 18
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                      left_margin=2, right_margin=3, closure_type=Closure(2),
                                                                      num_op_output=Operation(1), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=0, right_margin=3,
                                                         closure_type=Closure(0), num_op_output=Operation(1), axis_param=0,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 20
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0, right_margin=3, closure_type=Closure(2),
                                                                        num_op_output=Operation(2), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=0, right_margin=3,
                                                         closure_type=Closure(2), num_op_output=Operation(2),
                                                         axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 20 Failed: Expected False, but got True"
        print_and_log("Test Case 20 Passed: Expected False, got False")

        # Caso 21
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                        left_margin=0, right_margin=50, closure_type=Closure(2),
                                                                        num_op_output=Operation(2), axis_param=1)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=0, right_margin=3,
                                                         closure_type=Closure(2), num_op_output=Operation(2), axis_param=1,
                                                         belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                         data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 21 Failed: Expected True, but got False"
        print_and_log("Test Case 21 Passed: Expected True, got True")

        # Caso 22
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4, closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=None)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, got False")

        # Caso 23
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4, closure_type=Closure(0),
                                                                       num_op_output=Operation(3), axis_param=0)

        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3), axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                           left_margin=2, right_margin=4,
                                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                                           axis_param=None, field_in=None, field_out=None)
        print_and_log("Test Case 24.1 Passed: expected ValueError, got ValueError")
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=None, field_out=None,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 24.2 Passed: expected ValueError, got ValueError")

        # Caso 25
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3), num_op_output=Operation(0),
                                                                       axis_param=None, field_in=field_in, field_out=field_out)

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4,
                                                                       closure_type=Closure(3), num_op_output=Operation(1),
                                                                       axis_param=None, field_in=field_in, field_out=field_out)
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 26 Failed: Expected False, but got True"
        print_and_log("Test Case 26 Passed: Expected False, got False")

        # Caso 27
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4, closure_type=Closure(3),
                                                                       num_op_output=Operation(2), axis_param=None,
                                                                       field_in=field_in, field_out=field_out)

        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        field_in = 'track_popularity'
        field_out = 'track_popularity'
        expected = self.data_transformations.transform_interval_num_op(data_dictionary=self.rest_of_dataset.copy(),
                                                                       left_margin=2, right_margin=4, closure_type=Closure(3),
                                                                       num_op_output=Operation(3), axis_param=None, field_in=field_in,
                                                                       field_out=field_out)

        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=self.rest_of_dataset.copy(), left_margin=2, right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 28 Failed: Expected False, but got True"
        print_and_log("Test Case 28 Passed: Expected False, got False")


    def execute_checkInv_SpecialValue_FixValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_FixValue
        """
        print_and_log("Testing checkInv_SpecialValue_FixValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_FixValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_FixValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_SpecialValue_FixValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_FixValue
        """
        # Caso 1 - Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2 - Ejecutar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3 - Ejecutar invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = 10.5
        result_df = self.data_transformations.transform_special_value_fix_value(data_dictionary=self.small_batch_dataset.copy(),
                                                                                special_type_input=special_type_input,
                                                                                fix_value_output=fix_value_output,
                                                                                field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                             data_dictionary_out=result_df, special_type_input=special_type_input,
                                                                             missing_values=None, fix_value_output=fix_value_output,
                                                                             belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                             field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            fix_value_output=fix_value_output, belong_op_in=Belong(0),
            belong_op_out=Belong(0), axis_param=0)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar los valores invalid 1 y 3 en todas las columnas numéricas del batch pequeño
        # del dataset de prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3]
        fix_value_output = 101
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de todas las columnas numéricas
        # del batch pequeño del dataset de prueba por el valor 200. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = 200
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: cambiar los valores missing, así como el valor "Maroon 5" y "Katy Perry" de las columans de tipo string
        # del batch pequeño del dataset de prueba por el valor "SpecialValue". Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = ["Maroon 5", "Katy Perry"]
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Ejecutar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        fix_value_output = "SpecialValue"
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_fix_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_special_value_fix_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result, special_type_input=special_type_input,
                fix_value_output=fix_value_output,
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10 - Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 11 - Ejecutar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12 - Ejecutar invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

        # Caso 13
        # Ejecutar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=None, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            fix_value_output=fix_value_output, belong_op_in=Belong(0),
            belong_op_out=Belong(1), axis_param=0)
        assert invariant_result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")


    def execute_WholeDatasetTests_checkInv_SpecialValue_FixValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_FixValue
        """
        # Caso 1 - Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2 - Ejecutar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3 - Ejecutar invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = 10.4
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=None, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            fix_value_output=fix_value_output, belong_op_in=Belong(0),
            belong_op_out=Belong(0), axis_param=0)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar los valores invalid 1 y 3 en todas las columnas numéricas del batch pequeño
        # del dataset de prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3]
        fix_value_output = 101
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de todas las columnas numéricas
        # del batch pequeño del dataset de prueba por el valor 200. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = 200
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: cambiar los valores missing, así como el valor "Maroon 5" y "Katy Perry" de las columans de tipo string
        # del batch pequeño del dataset de prueba por el valor "SpecialValue". Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = ["Maroon 5", "Katy Perry"]
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Ejecutar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        fix_value_output = "SpecialValue"
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_fix_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_special_value_fix_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result, special_type_input=special_type_input,
                fix_value_output=fix_value_output,
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10 - Ejecutar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [0, -1]
        fix_value_output = "SpecialValue"
        field_in = 'instrumentalness'
        field_out = 'instrumentalness'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, and got False")

        # Caso 11 - Ejecutar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 3]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input, missing_values=missing_values,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, and got False")

        # Caso 12 - Ejecutar invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 6]
        fix_value_output = "SpecialValue"
        field_in = 'key'
        field_out = 'key'
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output, field_in=field_in, field_out=field_out)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=missing_values, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, and got False")

        # Caso 13
        # Ejecutar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            missing_values=None, fix_value_output=fix_value_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        fix_value_output = "SpecialValue"
        result_df = self.data_transformations.transform_special_value_fix_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            fix_value_output=fix_value_output, axis_param=0)

        invariant_result = self.invariants.check_inv_special_value_fix_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df, special_type_input=special_type_input,
            fix_value_output=fix_value_output, belong_op_in=Belong(0),
            belong_op_out=Belong(1), axis_param=0)
        assert invariant_result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")


    def execute_checkInv_SpecialValue_DerivedValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_DerivedValue
        """
        print_and_log("Testing checkInv_SpecialValue_DerivedValue Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field_in = 'acousticness'
        field_out = 'acousticness'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, missing_values=missing_values,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        field_out = 'acousticness'
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field_in].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values,
            axis_param=0, belong_op_in=Belong(0),
            belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar los valores faltantes 1, 3, 0.13 y 0.187, así como los valores nulos de python por el valor derivado 2 (Next) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1
        # (Previous) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                derived_type_output=derived_type_output,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                derived_type_output=derived_type_output,
                axis_param=None)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Ejecutar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        derived_type_output = DerivedType(0)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, and got True")

        # Caso 11
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, and got True")

        # Caso 12
        # Ejecutar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            derived_type_output=derived_type_output,
            axis_param=0)
        assert invariant_result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, and got True")

        # Caso 13
        # Ejecutar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field_in = 'acousticness'
        field_out = 'acousticness'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, missing_values=missing_values,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 13 Failed: Expected True, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        field_out = 'acousticness'
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field_in].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 14 Failed: Expected True, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")

        # Caso 15
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, and got False")

        # Caso 16
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, and got False")

        # Caso 17
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values,
            axis_param=0, belong_op_in=Belong(0),
            belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, and got False")

        # Caso 20 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.small_batch_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.small_batch_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                belong_op_in=Belong(0), belong_op_out=Belong(1),
                derived_type_output=derived_type_output,
                axis_param=None)
        print_and_log("Test Case 20 Passed: expected ValueError, got ValueError")


        # Caso 22
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, and got False")

        # Caso 23
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, and got False")

        # Caso 24
        # Ejecutar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            derived_type_output=derived_type_output,
            axis_param=0)
        assert invariant_result is False, "Test Case 24 Failed: Expected False, but got True"
        print_and_log("Test Case 24 Passed: Expected False, and got False")


    def execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_DerivedValue
        """
        # Caso 1
        # Ejecutar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field_in = 'acousticness'
        field_out = 'acousticness'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, missing_values=missing_values,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        field_out = 'acousticness'
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values,
            axis_param=0, belong_op_in=Belong(0),
            belong_op_out=Belong(0))
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: cambiar los valores faltantes 1, 3, 0.13 y 0.187, así como los valores nulos de python por el valor derivado 2 (Next) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1
        # (Previous) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                derived_type_output=derived_type_output,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                derived_type_output=derived_type_output,
                axis_param=None)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Ejecutar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        special_type_input = SpecialType(2)
        field_in = 'p'
        field_out = 'p'
        derived_type_output = DerivedType(0)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                field_in=field_in, field_out=field_out)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                belong_op_in=Belong(0), belong_op_out=Belong(0),
                field_in=field_in, field_out=field_out)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, and got True")

        # Caso 11
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, and got True")

        # Caso 12
        # Ejecutar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            derived_type_output=derived_type_output,
            axis_param=0)
        assert invariant_result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, and got True")

        # Caso 13
        # Ejecutar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field_in = 'acousticness'
        field_out = 'acousticness'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, missing_values=missing_values,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 13 Failed: Expected True, but got True"
        print_and_log("Test Case 13 Passed: Expected False, and got False")

        # Caso 14
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        field_in = 'acousticness'
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values, field_in=field_in, field_out=field_out,
            derived_type_output=derived_type_output)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out)
        assert invariant_result is False, "Test Case 14 Failed: Expected True, but got True"
        print_and_log("Test Case 14 Passed: Expected False, and got False")

        # Caso 15
        # Ejecutar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, and got False")

        # Caso 16
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values, axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            missing_values=missing_values,
            derived_type_output=derived_type_output,
            data_dictionary_out=result_df,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, and got False")

        # Caso 17
        # Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            axis_param=0, missing_values=missing_values)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            missing_values=missing_values,
            axis_param=0, belong_op_in=Belong(0),
            belong_op_out=Belong(1))
        assert invariant_result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, and got False")

        # Caso 20 - Ejecutar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derived_type_output = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.data_transformations.transform_special_value_derived_value(
                data_dictionary=self.rest_of_dataset.copy(),
                special_type_input=special_type_input,
                derived_type_output=derived_type_output,
                missing_values=missing_values, axis_param=None)
            invariant_result = self.invariants.check_inv_special_value_derived_value(
                data_dictionary_in=self.rest_of_dataset.copy(),
                data_dictionary_out=result,
                special_type_input=special_type_input,
                missing_values=missing_values,
                belong_op_in=Belong(0), belong_op_out=Belong(1),
                derived_type_output=derived_type_output,
                axis_param=None)
        print_and_log("Test Case 20 Passed: expected ValueError, got ValueError")


        # Caso 22
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(0)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            derived_type_output=derived_type_output,
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, and got False")

        # Caso 23
        # Ejecutar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        field_in = 'danceability'
        field_out = 'danceability'
        derived_type_output = DerivedType(1)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out, derived_type_output=derived_type_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, and got False")

        # Caso 24
        # Ejecutar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        derived_type_output = DerivedType(2)
        result_df = self.data_transformations.transform_special_value_derived_value(
            data_dictionary=self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            derived_type_output=derived_type_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_derived_value(
            data_dictionary_in=self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            derived_type_output=derived_type_output,
            axis_param=0)
        assert invariant_result is False, "Test Case 24 Failed: Expected False, but got True"
        print_and_log("Test Case 24 Passed: Expected False, and got False")


    def execute_checkInv_SpecialValue_NumOp(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_NumOp
        """
        print_and_log("Testing checkInv_SpecialValue_NumOp Data Transformation Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_NumOp()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_NumOp()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_SpecialValue_NumOp(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_NumOp
        """
        # MISSING
        # Caso 1
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")


        # Caso 2 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=None)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")


        # Caso 3 Se aplica la mediana de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")


        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")


        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto. Se calcula el closest sobre el dataframe entero en relación a los valores faltantes y valores nulos.
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=None)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")


        # Caso 6
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las
        # columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        expected_df_copy = expected_df.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=expected_df_copy,
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")


        # Caso 7
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=None)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")


        # Caso 8
        # Ejecutar la invariante: aplicar la media de cada columna numérica a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")


        # Caso 9
        # Ejecutar la invariante: aplicar la mediana de cada columna numérica del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, and got True")


        # Caso 10
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=None)
        assert invariant_result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, and got True")


        # Caso 11
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, and got True")


        # Caso 12
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=None)
        assert invariant_result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, and got True")


        # Caso 13
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'

        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, and got True")


        # Caso 14
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, and got True")


        # Caso 15
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, and got True")


        # Caso 16
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, and got True")


        # Caso 17
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, and got True")


        # Caso 18
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, and got True")


        # Caso 19
        # Ejecutar la invariante: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, and got True")

        # Caso 20
        # Ejecutar la invariante: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        print(result_df.loc[6, 'acousticness'])
        assert invariant_result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, and got True")

        # Caso 21
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, and got False")

        # Caso 22 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, and got False")

        # Caso 24
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 24 Failed: Expected False, but got True"
        print_and_log("Test Case 24 Passed: Expected False, and got False")


        # Caso 27
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, and got False")

        # Caso 30
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected False, and got False")

        # Caso 32
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 32 Failed: Expected False, but got True"
        print_and_log("Test Case 32 Passed: Expected False, and got False")

        # Caso 33
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 33 Failed: Expected False, but got True"
        print_and_log("Test Case 33 Passed: Expected False, and got False")

        # Caso 34
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 34 Failed: Expected False, but got True"
        print_and_log("Test Case 34 Passed: Expected False, and got False")

        # Caso 37
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 37 Failed: Expected False, but got True"
        print_and_log("Test Case 37 Passed: Expected False, and got False")

        # Caso 39
        # Ejecutar la invariante: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary=self.small_batch_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in=self.small_batch_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 39 Failed: Expected False, but got True"
        print_and_log("Test Case 39 Passed: Expected False, and got False")

        # Caso 40
        # Ejecutar la invariante: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary=self.small_batch_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                          axis_param=0)
        assert invariant_result is False, "Test Case 40 Failed: Expected False, but got True"
        print_and_log("Test Case 40 Passed: Expected False, and got False")


    def execute_WholeDatasetTests_checkInv_SpecialValue_NumOp(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_NumOp
        """
        # MISSING
        # Caso 1
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary= self.rest_of_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in= self.rest_of_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, and got True")

        # Caso 2 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary= self.rest_of_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, missing_values=missing_values,
                                                                             axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in= self.rest_of_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=None)
        assert invariant_result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, and got True")

        # Caso 3 Se aplica la mediana de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, and got True")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, and got True")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto. Se calcula el closest sobre el dataframe entero en relación a los valores faltantes y valores nulos.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, and got True")

        # Caso 6
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las
        # columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df =  self.rest_of_dataset.copy()
        expected_df_copy = expected_df.copy()
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=expected_df_copy,
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          missing_values=missing_values,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, and got True")

        # Caso 7
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, and got True")

        # Caso 8
        # Ejecutar la invariante: aplicar la media de cada columna numérica a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, and got True")

        # Caso 9
        # Ejecutar la invariante: aplicar la mediana de cada columna numérica del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, and got True")

        # Caso 10
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, and got True")

        # Caso 11
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, and got True")

        # Caso 12
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=None)
        assert invariant_result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, and got True")

        # Caso 13
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, and got True")

        # Caso 14
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, and got True")

        # Caso 15
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, and got True")

        # Caso 16
        # Ejecutar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            field_in=field_in, field_out=field_out,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            field_in=field_in, field_out=field_out,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, and got True")

        # Caso 17
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, and got True")

        # Caso 18
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            axis_param=0)
        assert invariant_result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, and got True")

        # Caso 19
        # Ejecutar la invariante: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, field_in=field_in, field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            belong_op_in=Belong(0), belong_op_out=Belong(0),
            field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, and got True")

        # Caso 20
        # Ejecutar la invariante: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary= self.rest_of_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in= self.rest_of_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                          axis_param=0)
        assert invariant_result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, and got True")

        # Caso 21
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=0)
        assert invariant_result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, and got False")

        # Caso 22 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        special_type_input = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(1)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 22 Failed: Expected False, but got True"
        print_and_log("Test Case 22 Passed: Expected False, and got False")

        # Caso 30
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(2)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected False, and got False")

        # Caso 32
        # Ejecutar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(
            data_dictionary= self.rest_of_dataset.copy(),
            special_type_input=special_type_input,
            num_op_output=num_op_output, missing_values=missing_values,
            axis_param=None)
        invariant_result = self.invariants.check_inv_special_value_num_op(
            data_dictionary_in= self.rest_of_dataset.copy(),
            data_dictionary_out=result_df,
            special_type_input=special_type_input,
            num_op_output=num_op_output,
            missing_values=missing_values,
            belong_op_in=Belong(0), belong_op_out=Belong(1),
            axis_param=None)
        assert invariant_result is False, "Test Case 32 Failed: Expected False, but got True"
        print_and_log("Test Case 32 Passed: Expected False, and got False")

        # Caso 33
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary= self.rest_of_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             num_op_output=num_op_output, field_in=field_in,
                                                                             field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in= self.rest_of_dataset.copy(),
                                                                        data_dictionary_out=result_df,
                                                                        special_type_input=special_type_input,
                                                                        num_op_output=num_op_output,
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 33 Failed: Expected False, but got True"
        print_and_log("Test Case 33 Passed: Expected False, and got False")

        # Caso 34
        # Ejecutar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(0)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary= self.rest_of_dataset.copy(),
                                                                            special_type_input=special_type_input,
                                                                            field_in=None, field_out=None,
                                                                            num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in= self.rest_of_dataset.copy(),
                                                                            data_dictionary_out=result_df,
                                                                            special_type_input=special_type_input,
                                                                            num_op_output=num_op_output,
                                                                            field_in=None, field_out=None,
                                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                            axis_param=0)
        assert invariant_result is False, "Test Case 34 Failed: Expected False, but got True"
        print_and_log("Test Case 34 Passed: Expected False, and got False")

        # Caso 37
        # Ejecutar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(2)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary= self.rest_of_dataset.copy(),
                                                                            special_type_input=special_type_input,
                                                                            num_op_output=num_op_output, field_in=field_in,
                                                                             field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in= self.rest_of_dataset.copy(),
                                                                        data_dictionary_out=result_df,
                                                                        special_type_input=special_type_input,
                                                                        num_op_output=num_op_output,
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 37 Failed: Expected False, but got True"
        print_and_log("Test Case 37 Passed: Expected False, and got False")

        # Caso 39
        # Ejecutar la invariante: aplicar el closest a los valores outliers de la columna 'danceability' del batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        field_in = 'danceability'
        field_out = 'danceability'
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary= self.rest_of_dataset.copy(),
                                                                            special_type_input=special_type_input,
                                                                            num_op_output=num_op_output, field_in=field_in,
                                                                            field_out=field_out, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in= self.rest_of_dataset.copy(),
                                                                        data_dictionary_out=result_df,
                                                                        special_type_input=special_type_input,
                                                                        num_op_output=num_op_output,
                                                                        belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                        field_in=field_in, field_out=field_out, axis_param=0)
        assert invariant_result is False, "Test Case 39 Failed: Expected False, but got True"
        print_and_log("Test Case 39 Passed: Expected False, and got False")

        # Caso 40
        # Ejecutar la invariante: aplicar el closest a los valores outliers de cada columna del batch pequeño del dataset
        # de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        special_type_input = SpecialType(2)
        num_op_output = Operation(3)
        result_df = self.data_transformations.transform_special_value_num_op(data_dictionary= self.rest_of_dataset.copy(),
                                                                             special_type_input=special_type_input,
                                                                             field_in=None, field_out=None,
                                                                             num_op_output=num_op_output, axis_param=0)
        invariant_result = self.invariants.check_inv_special_value_num_op(data_dictionary_in= self.rest_of_dataset.copy(),
                                                                          data_dictionary_out=result_df,
                                                                          special_type_input=special_type_input,
                                                                          num_op_output=num_op_output,
                                                                          field_in=None, field_out=None,
                                                                          belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                          axis_param=0)
        assert invariant_result is False, "Test Case 40 Failed: Expected False, but got True"
        print_and_log("Test Case 40 Passed: Expected False, and got False")


    def execute_checkInv_MissingValue_MissingValue(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_NumOp
        """
        print_and_log("Testing checkInv_SpecialValue_NumOp Data Transformation Function")
        print_and_log("")

        pd.options.mode.chained_assignment = None # Suppresses warnings related to modifying copies of dataframes

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_MissingValue_MissingValue()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_MissingValue_MissingValue()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")


    def execute_SmallBatchTests_checkInv_MissingValue_MissingValue(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_MissingValue_MissingValue
        """
        # Caso 1
        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                       data_dictionary_out=self.small_batch_dataset.copy(),
                                                                       belong_op_out=Belong(1), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        in_df['danceability'][1] = np.NaN
        expected_df['danceability'][0] = 5
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in=None,
                                                                       field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6
        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=self.small_batch_dataset.copy(),
                                                                        data_dictionary_out=self.small_batch_dataset.copy(),
                                                                        belong_op_out=Belong(1), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        in_df['danceability'][1] = np.NaN
        expected_df['danceability'][0] = 5
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        expected_df = self.small_batch_dataset.copy()
        in_df = self.small_batch_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"'d'
        print_and_log("Test Case 10 Passed: Expected False, got False")


    def execute_WholeDatasetTests_checkInv_MissingValue_MissingValue(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_MissingValue_MissingValue
        """
        # Caso 1
        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                       data_dictionary_out=self.rest_of_dataset.copy(),
                                                                       belong_op_out=Belong(1), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        in_df['danceability'][1] = np.NaN
        expected_df['danceability'][0] = 5
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6
        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=self.rest_of_dataset.copy(),
                                                                        data_dictionary_out=self.rest_of_dataset.copy(),
                                                                        belong_op_out=Belong(1), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        in_df['danceability'][1] = np.NaN
        expected_df['danceability'][0] = 5
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        expected_df = self.rest_of_dataset.copy()
        in_df = self.rest_of_dataset.copy()
        in_df['danceability'][0] = np.NaN
        expected_df['danceability'][0] = np.NaN
        expected_df['danceability'][1] = 5

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=in_df,
                                                                       data_dictionary_out=expected_df,
                                                                       belong_op_out=Belong(0), field_in='danceability',
                                                                       field_out='danceability')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"'d'
        print_and_log("Test Case 10 Passed: Expected False, got False")
