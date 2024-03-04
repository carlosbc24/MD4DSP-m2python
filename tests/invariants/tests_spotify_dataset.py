import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

from functions.contract_invariants import ContractsInvariants
from helpers.enumerations import Closure, DataType
from helpers.enumerations import DerivedType, Operation
from helpers.logger import print_and_log


# TODO: Implement the invariants tests with external dataset
class InvariantsExternalDatasetTests(unittest.TestCase):
    """
        Class to test the invariants with external dataset test cases

        Attributes:
        pre_post (ContractsPrePost): instance of the class ContractsPrePost
        dataDictionary (pd.DataFrame): dataframe with the external dataset. It must be loaded in the __init__ method

        Methods:
        executeAll_ExternalDatasetTests: execute all the invariants with external dataset tests
        execute_CheckInv_FixValue_FixValue_ExternalDatasetTests: execute the invariant test with external dataset for
        the function checkInv_FixValue_FixValue execute_SmallBatchTests_CheckInv_FixValue_FixValue_ExternalDataset:
        execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_FixValue
        execute_WholeDatasetTests_CheckInv_FixValue_FixValue_ExternalDataset: execute the invariant test using the
        whole dataset for the function checkInv_FixValue_FixValue
        execute_checkInv_FixValue_DerivedValue_ExternalDatasetTests: execute the invariant test with external dataset
        for the function checkInv_FixValue_DerivedValue
        execute_SmallBatchTests_checkInv_FixValue_DerivedValue_ExternalDataset: execute the invariant test using a
        small batch of the dataset for the function checkInv_FixValue_DerivedValue
        execute_WholeDatasetTests_checkInv_FixValue_DerivedValue_ExternalDataset: execute the invariant test using
        the whole dataset for the function checkInv_FixValue_DerivedValue
        execute_checkInv_FixValue_NumOp_ExternalDatasetTests: execute the invariant test with external dataset for
        the function checkInv_FixValue_NumOp execute_SmallBatchTests_checkInv_FixValue_NumOp_ExternalDataset: execute
        the invariant test using a small batch of the dataset for the function checkInv_FixValue_NumOp
        execute_WholeDatasetTests_checkInv_FixValue_NumOp_ExternalDataset: execute the invariant test using the whole
        dataset for the function checkInv_FixValue_NumOp execute_checkInv_Interval_FixValue_ExternalDatasetTests:
        execute the invariant test with external dataset for the function checkInv_Interval_FixValue
        execute_SmallBatchTests_checkInv_Interval_FixValue_ExternalDataset: execute the invariant test using a small
        batch of the dataset for the function checkInv_Interval_FixValue
        execute_WholeDatasetTests_checkInv_Interval_FixValue_ExternalDataset: execute the invariant test using the
        whole dataset for the function checkInv_Interval_FixValue
        execute_checkInv_Interval_DerivedValue_ExternalDatasetTests: execute the invariant test with external dataset
        for the function checkInv_Interval_DerivedValue
        execute_SmallBatchTests_checkInv_Interval_DerivedValue_ExternalDataset: execute the invariant test using a
        small batch of the dataset for the function checkInv_Interval_DerivedValue
        execute_WholeDatasetTests_checkInv_Interval_DerivedValue_ExternalDataset: execute the invariant test using
        the whole dataset for the function checkInv_Interval_DerivedValue
        execute_checkInv_Interval_NumOp_ExternalDatasetTests: execute the invariant test with external dataset for
        the function checkInv_Interval_NumOp execute_SmallBatchTests_checkInv_Interval_NumOp_ExternalDataset: execute
        the invariant test using a small batch of the dataset for the function checkInv_Interval_NumOp
        execute_WholeDatasetTests_checkInv_Interval_NumOp_ExternalDataset: execute the invariant test using the whole
        dataset for the function checkInv_Interval_NumOp execute_checkInv_SpecialValue_FixValue_ExternalDatasetTests:
        execute the invariant test with external dataset for the function checkInv_SpecialValue_FixValue
        execute_SmallBatchTests_checkInv_SpecialValue_FixValue_ExternalDataset: execute the invariant test using a
        small batch of the dataset for the function checkInv_SpecialValue_FixValue
        execute_WholeDatasetTests_checkInv_SpecialValue_FixValue_ExternalDataset: execute the invariant test using
        the whole dataset for the function checkInv_SpecialValue_FixValue
        execute_checkInv_SpecialValue_DerivedValue_ExternalDatasetTests: execute the invariant test with external
        dataset for the function checkInv_SpecialValue_DerivedValue
        execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue_ExternalDataset: execute the invariant test using
        a small batch of the dataset for the function checkInv_SpecialValue_DerivedValue
        execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue_ExternalDataset: execute the invariant test
        using the whole dataset for the function checkInv_SpecialValue_DerivedValue
        execute_checkInv_SpecialValue_NumOp_ExternalDatasetTests: execute the invariant test with external dataset
        for the function checkInv_SpecialValue_NumOp
        execute_SmallBatchTests_checkInv_SpecialValue_NumOp_ExternalDataset: execute the invariant test using a small
        batch of the dataset for the function checkInv_SpecialValue_NumOp
        execute_WholeDatasetTests_checkInv_SpecialValue_NumOp_ExternalDataset: execute the invariant test using the
        whole dataset for the function checkInv_SpecialValue_NumOp
    """

    def __init__(self):
        """
        Constructor of the class

        Atributes:
        invariants (ContractsInvariants): instance of the class ContractsInvariants
        data_dictionary (pd.DataFrame): dataframe with the external dataset. It must be loaded in the __init__ method
        """
        self.invariants = ContractsInvariants()

        # Get the current directory
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the CSV file
        ruta_csv = os.path.join(directorio_actual, '../../test_datasets/spotify_songs/spotify_songs.csv')
        # Create the dataframe with the external dataset
        self.data_dictionary = pd.read_csv(ruta_csv)

        # Select a small batch of the dataset (first 10 rows)
        self.small_batch_dataset = self.data_dictionary.head(10)
        # Select the rest of the dataset (from row 11 to the end)
        self.rest_of_dataset = self.data_dictionary.iloc[10:]

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

        for test_method in tqdm(test_methods, desc="Running Invariant Contracts Simple Tests", unit="test"):
            test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ DATASET INVARIANT TEST CASES FINISHED -----")
        print_and_log("--------------------------------------------------")
        print_and_log("")

    def execute_CheckInv_FixValue_FixValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_FixValue
        """
        print_and_log("Testing checkInv_FixValue_FixValue Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_CheckInv_FixValue_FixValue_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_CheckInv_FixValue_FixValue_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_CheckInv_FixValue_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_FixValue
        """

        # Caso 1
        # Comprobar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 67
        fixValueOutput = 1
        field = 'track_popularity'
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput, field=field)
        expected_df['track_popularity'] = expected_df['track_popularity'].replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 'All the Time - Don Diablo Remix'
        fixValueOutput = 'todos los tiempo - Don Diablo Remix'
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df = expected_df.replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch pequeño del dataset de prueba. Sobre un dataframe de
        # copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = '2019-07-05'
        fixValueOutput = True
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df['track_album_release_date'] = expected_df['track_album_release_date'].replace(fixValueInput,
                                                                                                  fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor fijo string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 'Maroon 5'
        fixValueOutput = 3.0
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df = expected_df.replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar el valor fijo de tipo FLOAT 2.33e-5 por el valor fijo de tipo STRING "Near 0"
        # sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 2.33e-5
        fixValueOutput = "Near 0"
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df = expected_df.replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

    def execute_WholeDatasetTests_CheckInv_FixValue_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_FixValue
        """

        # Caso 1
        # Comprobar la invariante: cambiar el valor fijo 67 de la columna track_popularity por el valor fijo 1 sobre
        # el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 67
        fixValueOutput = 1
        field = 'track_popularity'
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput, field=field)
        expected_df['track_popularity'] = expected_df['track_popularity'].replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor fijo 'All the Time - Don Diablo Remix'
        # del dataframe por el valor fijo 'todos los tiempo - Don Diablo Remix' sobre el batch grande del dataset de
        # prueba. Sobre un dataframe de copia del batch grande del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado. En este caso se prueba sobre el dataframe entero
        # independientemente de la columna
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 'All the Time - Don Diablo Remix'
        fixValueOutput = 'todos los tiempo - Don Diablo Remix'
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df = expected_df.replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch grande del dataset de prueba. Sobre un dataframe de
        # copia del batch grande del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = '2019-07-05'
        fixValueOutput = True
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df['track_album_release_date'] = expected_df['track_album_release_date'].replace(fixValueInput,
                                                                                                  fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor fijo string 'Maroon 5' por el valor fijo de tipo FLOAT 3.0
        # sobre el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 'Maroon 5'
        fixValueOutput = 3.0
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df = expected_df.replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar el valor fijo de tipo FLOAT 2.33e-5 por el valor fijo de tipo STRING "Near 0"
        # sobre el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 2.33e-5
        fixValueOutput = "Near 0"
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df = expected_df.replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar el valor fijo de tipo FLOAT 0.833, presente en varias columnas del dataframe,
        # por el valor fijo de tipo entero 1 sobre el batch
        # grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba cambiar los
        # valores manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.833
        fixValueOutput = 1
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df = expected_df.replace(fixValueInput, fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

    def execute_checkInv_FixValue_DerivedValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_DerivedValue
        """
        print_and_log("Testing checkInv_FixValue_DerivedValue Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_FixValue_DerivedValue_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_FixValue_DerivedValue_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_FixValue_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_DerivedValue
        """

        # Caso 1
        # Comprobar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0
        field = 'mode'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(0), field=field)
        most_frequent_mode_value = expected_df['mode'].mode()[0]
        expected_df['mode'] = expected_df['mode'].replace(fixValueInput, most_frequent_mode_value)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 'Katy Perry'
        field = 'track_artist'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(1), field=field)
        # Sustituir el valor fijo definido por la variable 'fixValueInput' del dataframe expected por el valor previo a nivel de columna, es deicr, el valor en la misma columna pero en la fila anterior
        # Identificar índices donde 'Katy Perry' es el valor en la columna 'track_artist'.
        katy_perry_indices = expected_df.loc[expected_df[field] == fixValueInput].index

        # Iterar sobre los índices y reemplazar cada 'Katy Perry' por el valor previo en la columna.
        for idx in katy_perry_indices[::-1]:
            if idx > 0:  # Asegura que no esté intentando acceder a un índice fuera de rango.
                expected_df.at[idx, field] = expected_df.at[idx - 1, field]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el batch pequeño
        # del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = '2019-12-13'
        field = 'track_album_release_date'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(2), field=field)
        date_indices = expected_df.loc[expected_df[field] == fixValueInput].index
        for idx in date_indices:
            if idx < len(expected_df) - 1:
                expected_df.at[idx, field] = expected_df.at[idx + 1, field]
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Inavriante. cambair el valor fijo 'pop' de la columna 'playlist_genre' por el valor derivado 2 (next)
        # a nivel de fila sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 'pop'
        field = 'playlist_genre'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(2), axis_param=1)
        pop_indices = expected_df.loc[expected_df[field] == fixValueInput].index
        # Gets next column name to 'field' in the dataframe
        next_column = expected_df.columns[expected_df.columns.get_loc(field) + 1]
        for idx in pop_indices:
            if next_column in expected_df.columns:
                expected_df.at[idx, field] = expected_df.at[idx, next_column]
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Check the invariant: chenged the fixed value 0.11 of the column 'liveness'
        # by the derived value 0 (most frequent) at column level on the small batch of the test dataset.
        # On a copy of the small batch of the test dataset, change the values manually and verify if the result matches
        # the expected
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0.11
        field = 'liveness'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(0), field=field)
        most_frequent_liveness_value = expected_df[field].mode()[0]
        expected_df[field] = expected_df[field].replace(fixValueInput, most_frequent_liveness_value)
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Check the invariant: chenged the fixed value 'Ed Sheeran' of all the dataset by the most frequent value
        # from the small batch dataset. On a copy of the small batch of the test dataset, change the values manually and
        # verify if the result matches the expected
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 'Ed Sheeran'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(0))
        # Get most frequent value of the all columns from small batch dataset

        # Convertir el DataFrame a una Serie para contar los valores en todas las columnas
        all_values = expected_df.melt(value_name="values")['values']
        # Obtener el valor más frecuente
        most_frequent_value = all_values.value_counts().idxmax()
        expected_df = expected_df.replace(fixValueInput, most_frequent_value)
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Invariante: exception after trying to gte previous value from all the dataset without specifying the column or
        # row level
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 'pop'
        field = 'playlist_genre'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                           fixValueInput=fixValueInput,
                                                           derivedTypeOutput=DerivedType(1))
        print_and_log("Test Case 7 Passed: the function raised the expected exception")

        # Caso 8: exception after trying to get the next value from all the dataset without specifying the
        # column or row level
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 'pop'
        field = 'playlist_genre'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                           fixValueInput=fixValueInput,
                                                           derivedTypeOutput=DerivedType(2))
        print_and_log("Test Case 8 Passed: the function raised the expected exception")

        # Caso 9: exception after trying to get the previous value using a column level. The field doens't exist in the
        # dataset.
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 'pop'
        field = 'autor_artista'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset,
                                                           fixValueInput=fixValueInput,
                                                           derivedTypeOutput=DerivedType(1), field=field)
        print_and_log("Test Case 9 Passed: the function raised the expected exception")

    def execute_WholeDatasetTests_checkInv_FixValue_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_DerivedValue
        """
        # Caso 1
        # Comprobar la invariante: cambiar el valor fijo 0 de la columna 'mode' por el valor derivado 0 (Most frequent)
        # a nivel de columna sobre el resto del dataset. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        # Definir el resultado esperado
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0
        field = 'mode'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(0), field=field)
        most_frequent_mode_value = expected_df['mode'].mode()[0]
        expected_df['mode'] = expected_df['mode'].replace(fixValueInput, most_frequent_mode_value)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor fijo 'Katy Perry' de la columna 'artist_name'
        # por el valor derivado 2 (Previous) a nivel de columna sobre el resto del dataset.
        # Sobre un dataframe de copia del resto del dataset cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado
        # Get rows from 2228 to 2240
        subdataframe_2200 = self.data_dictionary.iloc[2227:2240]

        expected_df = subdataframe_2200.copy()
        fixValueInput = 'Katy Perry'
        field = 'track_artist'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(subdataframe_2200,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(1), field=field)
        # Sustituir el valor fijo definido por la variable 'fixValueInput' del dataframe expected por el valor previo
        # a nivel de columna, es deicr, el valor en la misma columna pero en la fila anterior Identificar índices
        # donde 'Katy Perry' es el valor en la columna 'track_artist'.
        # Identificar índices donde 'Katy Perry' es el valor en la columna 'field'.
        katy_perry_indices = expected_df.loc[expected_df[field] == fixValueInput].index

        # Iterar sobre los indices desde el ultimo hasta el primero, iterarlos inversamente
        for idx in katy_perry_indices[::-1]:
            if idx > 0:
                expected_df.at[idx, field] = expected_df.at[idx - 1, field]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor fijo de tipo fecha 2019-12-13 de
        # la columna 'track_album_release_date' por el valor derivado 3 (Next) a nivel de columna sobre el resto del dataset.
        # Sobre un dataframe de copia del resto del dataset cambiar los valores
        # manualmente y verificar si el resultado obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = '2019-12-13'
        field = 'track_album_release_date'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(2), field=field)
        date_indices = expected_df.loc[expected_df[field] == fixValueInput].index
        for idx in date_indices:
            if idx < len(expected_df) - 1:
                expected_df.at[idx, field] = expected_df.at[idx + 1, field]
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Inavriante. cambair el valor fijo 'pop' de la columna 'playlist_genre' por el valor derivado 2 (next)
        # a nivel de fila sobre el resto del dataset. Sobre un dataframe de copia del resto del dataset cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 'pop'
        field = 'playlist_genre'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(2), axis_param=1)
        pop_indices = expected_df.loc[expected_df[field] == fixValueInput].index
        # Gets next column name to 'field' in the dataframe
        next_column = expected_df.columns[expected_df.columns.get_loc(field) + 1]
        for idx in pop_indices:
            if next_column in expected_df.columns:
                expected_df.at[idx, field] = expected_df.at[idx, next_column]
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Check the invariant: chenged the fixed value 0.11 of the column 'liveness'
        # by the derived value 0 (most frequent) at column level on the rest of the dataset.
        # On a copy of the rest of the dataset, change the values manually and verify if the result matches
        # the expected
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.11
        field = 'liveness'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(0), field=field)
        most_frequent_liveness_value = expected_df[field].mode()[0]
        expected_df[field] = expected_df[field].replace(fixValueInput, most_frequent_liveness_value)
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Check the invariant: chenged the fixed value 'Ed Sheeran' of all the dataset by the most frequent value
        # from the rest of the dataset. On a copy of the rest of the dataset, change the values manually and
        # verify if the result matches the expected
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 'Ed Sheeran'
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset,
                                                                   fixValueInput=fixValueInput,
                                                                   derivedTypeOutput=DerivedType(0))
        # Get most frequent value of the all columns from the rest of the dataset

        # Convertir el DataFrame a una Serie para contar los valores en todas las columnas
        all_values = expected_df.melt(value_name="values")['values']
        # Obtener el valor más frecuente
        most_frequent_value = all_values.value_counts().idxmax()
        expected_df = expected_df.replace(fixValueInput, most_frequent_value)
        # Verify if the result matches the expected
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Invariante: exception after trying to gte previous value from all the dataset without specifying the column or
        # row level
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 'pop'
        field = 'playlist_genre'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset,
                                                           fixValueInput=fixValueInput,
                                                           derivedTypeOutput=DerivedType(1))
        print_and_log("Test Case 7 Passed: the function raised the expected exception")

        # Caso 8: exception after trying to get the next value from all the dataset without specifying the
        # column or row level
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 'pop'
        field = 'playlist_genre'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset,
                                                           fixValueInput=fixValueInput,
                                                           derivedTypeOutput=DerivedType(2))
        print_and_log("Test Case 8 Passed: the function raised the expected exception")

        # Caso 9: exception after trying to get the previous value using a column level. The field doens't exist in the
        # dataset.
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 'pop'
        field = 'autor_artista'
        expected_exception = ValueError
        # Verify if the result matches the expected
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset,
                                                           fixValueInput=fixValueInput,
                                                           derivedTypeOutput=DerivedType(1), field=field)
        print_and_log("Test Case 9 Passed: the function raised the expected exception")

    def execute_checkInv_FixValue_NumOp_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_FixValue_NumOp
        """
        print_and_log("Testing checkInv_FixValue_NumOp Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_FixValue_NumOp_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_FixValue_NumOp_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_FixValue_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_NumOp
        """
        # Caso 1
        # Comprobar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0
        field = 'instrumentalness'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(0), field=field)
        # Sustituir los valores 0 por los NaN en la columa field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df[field] = expected_df[field].interpolate(method='linear', limit_direction='both')
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0.725
        field = 'valence'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(1), field=field)
        # Sustituir el valor 0.725 por el valor de la media en la columna field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, expected_df[field].mean())
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 8
        field = 'key'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(2), field=field)
        # Sustituir el valor 8 por el valor de la mediana en la columna field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, expected_df[field].median())
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 99.972
        field = 'tempo'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(3), field=field)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Inicializar variables para almacenar el valor más cercano y la diferencia mínima
        closest_value = None
        min_diff = np.inf
        # Iterar sobre cada columna numérica para encontrar el valor más cercano a fixValueInput
        for col in numeric_columns:
            # Calcular la diferencia absoluta con fixValueInput y excluir el propio fixValueInput
            diff = expected_df[col].apply(lambda x: abs(x - fixValueInput) if x != fixValueInput else np.inf)
            # Encontrar el índice del valor mínimo que no sea el propio fixValueInput
            idx_min = diff.idxmin()
            # Comparar si esta diferencia es la más pequeña hasta ahora y actualizar closest_value y min_diff según sea necesario
            if diff[idx_min] < min_diff:
                closest_value = expected_df.at[idx_min, col]
                min_diff = diff[idx_min]
        # Sustituir el valor 99.972 por el valor más cercano en la columna field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, closest_value)

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")
        print(expected_df['tempo'])
        print(result_df['tempo'])

        # Caso 5
        # Comprobar la invariante: cambiar el valor 0 en todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 1 (media) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(1), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Calcula la media de todas las columnas numéricas
        mean_values = expected_df[numeric_columns].mean()
        # Sustituir todos los valores 0 por el valor de la media de aquellas columnas numéricas en expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fixValueInput, mean_values)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 2 (mediana) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0.65
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(2), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Calcula la mediana de todas las columnas numéricas
        median_values = expected_df[numeric_columns].median()
        # Sustituir todos los valores 0.65 por el valor de la mediana de aquellas columnas numéricas en expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fixValueInput, median_values)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 3 (más cercano) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0.65
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(3), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Inicializar variables para almacenar el valor más cercano y la diferencia mínima
        closest_value = None
        min_diff = np.inf
        # Iterar sobre cada columna numérica para encontrar el valor más cercano a fixValueInput
        for col in numeric_columns:
            # Calcular la diferencia absoluta con fixValueInput y excluir el propio fixValueInput
            diff = expected_df[col].apply(lambda x: abs(x - fixValueInput) if x != fixValueInput else np.inf)
            # Encontrar el índice del valor mínimo que no sea el propio fixValueInput
            idx_min = diff.idxmin()
            # Comparar si esta diferencia es la más pequeña hasta ahora y actualizar closest_value y min_diff según sea necesario
            if diff[idx_min] < min_diff:
                closest_value = expected_df.at[idx_min, col]
                min_diff = diff[idx_min]
        # Sustituir el valor 99.972 por el valor más cercano en la columna field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, closest_value)

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 0 (interpolación) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0.65
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(0), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Susituir los valores 0.65 por los NaN en las columnas numéricas de expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fixValueInput, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df[numeric_columns] = expected_df[numeric_columns].interpolate(method='linear', limit_direction='both')
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field=None y se lanza una excepción ValueError
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                    fixValueInput=fixValueInput,
                                                    numOpOutput=Operation(0))
        print_and_log("Test Case 9 Passed: the function raised the expected exception")

        # Caso 10
        # Comprobar la excepción: se pasa una columna que no existe
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0.65
        field = 'p'
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset,
                                                    fixValueInput=fixValueInput,
                                                    numOpOutput=Operation(0), field=field)
        print_and_log("Test Case 10 Passed: the function raised the expected exception")

    def execute_WholeDatasetTests_checkInv_FixValue_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_NumOp
        """
        # Caso 1
        # Comprobar la invariante: cambiar el valor 0 de la columna 'instrumentalness' por el valor de operación 0,
        # es decir, la interpolación a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe
        # de copia el batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0
        field = 'instrumentalness'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(0), field=field)
        # Sustituir los valores 0 por los NaN en la columa field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df[field] = expected_df[field].interpolate(method='linear', limit_direction='both')
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor 0.725 de la columna 'valence' por el valor de operación 1 (Mean),
        # es decir, la media a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.725
        field = 'valence'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(1), field=field)
        # Sustituir el valor 0.725 por el valor de la media en la columna field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, expected_df[field].mean())
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor 8 de la columna 'key' por el valor de operación 2 (Median),
        # es decir, la mediana a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset se prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 8
        field = 'key'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(2), field=field)
        # Sustituir el valor 8 por el valor de la mediana en la columna field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, expected_df[field].median())
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor 99.972 de la columna 'tempo' por el valor de operación 3 (Closest),
        # es decir, el valor más cercano a nivel de columna sobre el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 99.972
        field = 'tempo'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(3), field=field)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Inicializar variables para almacenar el valor más cercano y la diferencia mínima
        closest_value = None
        min_diff = np.inf
        # Iterar sobre cada columna numérica para encontrar el valor más cercano a fixValueInput
        for col in numeric_columns:
            # Calcular la diferencia absoluta con fixValueInput y excluir el propio fixValueInput
            diff = expected_df[col].apply(lambda x: abs(x - fixValueInput) if x != fixValueInput else np.inf)
            # Encontrar el índice del valor mínimo que no sea el propio fixValueInput
            idx_min = diff.idxmin()
            # Comparar si esta diferencia es la más pequeña hasta ahora y actualizar closest_value y min_diff según sea necesario
            if diff[idx_min] < min_diff:
                closest_value = expected_df.at[idx_min, col]
                min_diff = diff[idx_min]
        # Sustituir el valor 99.972 por el valor más cercano en la columna field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, closest_value)

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar el valor 0 en todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 1 (media) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(1), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Calcula la media de todas las columnas numéricas
        mean_values = expected_df[numeric_columns].mean()
        # Sustituir todos los valores 0 por el valor de la media de aquellas columnas numéricas en expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fixValueInput, mean_values)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 2 (mediana) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.65
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(2), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Calcula la mediana de todas las columnas numéricas
        median_values = expected_df[numeric_columns].median()
        # Sustituir todos los valores 0.65 por el valor de la mediana de aquellas columnas numéricas en expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fixValueInput, median_values)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 3 (más cercano) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.65
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(3), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Inicializar variables para almacenar el valor más cercano y la diferencia mínima
        closest_value = None
        min_diff = np.inf
        # Iterar sobre cada columna numérica para encontrar el valor más cercano a fixValueInput
        for col in numeric_columns:
            # Calcular la diferencia absoluta con fixValueInput y excluir el propio fixValueInput
            diff = expected_df[col].apply(lambda x: abs(x - fixValueInput) if x != fixValueInput else np.inf)
            # Encontrar el índice del valor mínimo que no sea el propio fixValueInput
            idx_min = diff.idxmin()
            # Comparar si esta diferencia es la más pequeña hasta ahora y actualizar closest_value y min_diff según
            # sea necesario
            if diff[idx_min] < min_diff:
                closest_value = expected_df.at[idx_min, col]
                min_diff = diff[idx_min]
        # Sustituir el valor 99.972 por el valor más cercano en la columna field de expected_df
        expected_df[field] = expected_df[field].replace(fixValueInput, closest_value)

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: cambiar el valor 0.65 de todas las columnas del batch pequeño del dataset de prueba por
        # el valor de operación 0 (interpolación) a nivel de columna. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el
        # esperado.
        # Crear un DataFrame de prueba
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.65
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(0), axis_param=0)
        # Seleccionar solo las columnas numéricas del dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Susituir los valores 0.65 por los NaN en las columnas numéricas de expected_df
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(fixValueInput, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df[numeric_columns] = expected_df[numeric_columns].interpolate(method='linear', limit_direction='both')
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field=None y se lanza una excepción ValueError
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                    fixValueInput=fixValueInput,
                                                    numOpOutput=Operation(0))
        print_and_log("Test Case 9 Passed: the function raised the expected exception")

        # Caso 10
        # Comprobar la excepción: se pasa una columna que no existe
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.65
        field = 'p'
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset,
                                                    fixValueInput=fixValueInput,
                                                    numOpOutput=Operation(0), field=field)
        print_and_log("Test Case 10 Passed: the function raised the expected exception")

    def execute_checkInv_Interval_FixValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_FixValue
        """
        print_and_log("Testing checkInv_Interval_FixValue Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_FixValue_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_FixValue_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_Interval_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_Interval_FixValue
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        field = 'track_popularity'

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset, leftMargin=65,
                                                            rightMargin=69, closureType=Closure(3),
                                                            dataTypeOutput=DataType(0), fixValueOutput='65<=Pop<=69',
                                                            field=field)

        expected_df['track_popularity'] = expected_df['track_popularity'].apply(
            lambda x: '65<=Pop<=69' if 65 <= x <= 69 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset, leftMargin=0.5,
                                                            rightMargin=1, closureType=Closure(0), fixValueOutput=2)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: 2 if (np.issubdtype(type(x), np.number) and
                                                                              ((x > 0.5) and (x < 1))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        field = 'speechiness'

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset, leftMargin=0.06,
                                                            rightMargin=0.1270, closureType=Closure(1),
                                                            fixValueOutput=33, field=field)

        expected_df['speechiness'] = expected_df['speechiness'].apply(lambda x: 33 if 0.06 < x <= 0.1270 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        field = 'p'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset, leftMargin=65,
                                                                rightMargin=69, closureType=Closure(2),
                                                                fixValueOutput=101, field=field)
        print_and_log("Test Case 4 Passed: expected ValueError, got ValueError")

    def execute_WholeDatasetTests_checkInv_Interval_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_FixValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field = 'track_popularity'

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.rest_of_dataset, leftMargin=65,
                                                            rightMargin=69, closureType=Closure(2),
                                                            dataTypeOutput=DataType(0), fixValueOutput='65<=Pop<69',
                                                            field=field)

        expected_df['track_popularity'] = expected_df['track_popularity'].apply(
            lambda x: '65<=Pop<69' if 65 <= x < 69 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.rest_of_dataset, leftMargin=0.5,
                                                            rightMargin=1, closureType=Closure(0), fixValueOutput=2)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: 2 if (np.issubdtype(type(x), np.number) and
                                                                              ((x > 0.5) and (x < 1))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        field = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.rest_of_dataset, leftMargin=65,
                                                                rightMargin=69, closureType=Closure(2),
                                                                fixValueOutput=101, field=field)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

    def execute_checkInv_Interval_DerivedValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_DerivedValue
        """
        print_and_log("Testing checkInv_Interval_DerivedValue Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_DerivedValue_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_DerivedValue_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_SmallBatchTests_checkInv_Interval_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        field = 'liveness'

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(0), field=field)

        most_frequent_value = expected_df[field].mode().iloc[0]
        expected_df[field] = expected_df[field].apply(lambda x: most_frequent_value if 0.2 < x < 0.4 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(1), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i - 1] if np.issubdtype(
                                                                          type(value),
                                                                          np.number) and 0.2 < value < 0.4 and i > 0
                                                                      else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i + 1] if np.issubdtype(
                                                                          type(value), np.number) and 0.2 < value < 0.4
                                                                                                and i < len(
                                                                          row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset,
                                                                    leftMargin=0.2, rightMargin=0.4,
                                                                    closureType=Closure(0),
                                                                    derivedTypeOutput=DerivedType(2), axis_param=None)
        print_and_log("Test Case 4 Passed: expected ValueError, got ValueError")

    def execute_WholeDatasetTests_checkInv_Interval_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field = 'liveness'

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(0), field=field)

        most_frequent_value = expected_df[field].mode().iloc[0]
        expected_df[field] = expected_df[field].apply(lambda x: most_frequent_value if 0.2 < x < 0.4 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(1), axis_param=1)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i - 1] if np.issubdtype(
                                                                          type(value),
                                                                          np.number) and 0.2 < value < 0.4 and i > 0
                                                                      else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index), axis=1)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(0), axis_param=None)

        most_frequent_value = expected_df.stack().value_counts().idxmax()
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: most_frequent_value if np.issubdtype(type(x), np.number)
                                                                   and 0.2 < x < 0.4 else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i + 1] if np.issubdtype(
                                                                          type(value), np.number) and 0.2 < value < 0.4
                                                                                                and i < len(
                                                                          row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

    def execute_checkInv_Interval_NumOp_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_Interval_NumOp
        """
        print_and_log("Testing checkInv_Interval_NumOp Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_Interval_NumOp_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_Interval_NumOp_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_SmallBatchTests_checkInv_Interval_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_Interval_NumOp
        """

    # TODO: Implement the invariant tests with external dataset
    def execute_WholeDatasetTests_checkInv_Interval_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_NumOp
        """

    def execute_checkInv_SpecialValue_FixValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_FixValue
        """
        print_and_log("Testing checkInv_SpecialValue_FixValue Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_FixValue_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_FixValue_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_SmallBatchTests_checkInv_SpecialValue_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_FixValue
        """

    # TODO: Implement the invariant tests with external dataset
    def execute_WholeDatasetTests_checkInv_SpecialValue_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_FixValue
        """

    def execute_checkInv_SpecialValue_DerivedValue_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_DerivedValue
        """
        print_and_log("Testing checkInv_SpecialValue_DerivedValue Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_DerivedValue
        """

    # TODO: Implement the invariant tests with external dataset
    def execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_DerivedValue
        """

    def execute_checkInv_SpecialValue_NumOp_ExternalDatasetTests(self):
        """
        Execute the invariant test with external dataset for the function checkInv_SpecialValue_NumOp
        """
        print_and_log("Testing checkInv_SpecialValue_NumOp Invariant Function")
        print_and_log("")

        print_and_log("Dataset tests using small batch of the dataset:")
        self.execute_SmallBatchTests_checkInv_SpecialValue_NumOp_ExternalDataset()
        print_and_log("")
        print_and_log("Dataset tests using the whole dataset:")
        self.execute_WholeDatasetTests_checkInv_SpecialValue_NumOp_ExternalDataset()

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    # TODO: Implement the invariant tests with external dataset
    def execute_SmallBatchTests_checkInv_SpecialValue_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_NumOp
        """

    # TODO: Implement the invariant tests with external dataset
    def execute_WholeDatasetTests_checkInv_SpecialValue_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_NumOp
        """
