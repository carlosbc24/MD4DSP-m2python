import os
import time
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

from functions.contract_invariants import ContractsInvariants
from helpers.auxiliar import find_closest_value
from helpers.enumerations import Closure, DataType, SpecialType
from helpers.enumerations import DerivedType, Operation
from helpers.logger import print_and_log


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

        for test_method in tqdm(test_methods, desc="Running Invariant Contracts External Dataset Tests", unit="test"):
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
            self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
            self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
            self.invariants.checkInv_FixValue_DerivedValue(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset.copy(),
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
            self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset.copy(),
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
            self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset.copy(),
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
            self.invariants.checkInv_FixValue_DerivedValue(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(0), field=field)
        expected_df_copy = expected_df.copy()
        # Sustituir los valores 0 por los NaN en la columa field de expected_df
        expected_df_copy[field] = expected_df_copy[field].replace(fixValueInput, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df_copy[field] = expected_df_copy[field].interpolate(method='linear', limit_direction='both')
        # Para cada índice en la columna
        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field]):
                # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                expected_df_copy.at[idx, field] = expected_df.at[idx, field]
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df_copy)
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
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
        expected_df = self.small_batch_dataset.copy()
        fixValueInput = 0
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(3), axis_param=0)
        # Seleccionar solo las columnas numéricas del sub-DataFrame expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns

        # Iterar sobre cada columna numérica para encontrar y reemplazar el valor más cercano a fixValueInput
        for col in numeric_columns:
            # Inicializar variables para almacenar el valor más cercano y la diferencia mínima para cada columna
            closest_value = None
            min_diff = np.inf

            # Calcular la diferencia absoluta con fixValueInput y excluir el propio fixValueInput
            diff = expected_df[col].apply(lambda x: abs(x - fixValueInput) if x != fixValueInput else np.inf)

            # Encontrar el índice del valor mínimo que no sea el propio fixValueInput
            idx_min = diff.idxmin()

            # Comparar si esta diferencia es la más pequeña hasta ahora y actualizar closest_value y min_diff según sea necesario
            if diff[idx_min] < min_diff:
                closest_value = expected_df.at[idx_min, col]
                min_diff = diff[idx_min]

            # Sustituir el valor fixValueInput por el valor más cercano encontrado en la misma columna del sub-DataFrame expected_df
            expected_df[col] = expected_df[col].replace(fixValueInput, closest_value)

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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
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
            self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
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
            self.invariants.checkInv_FixValue_NumOp(self.small_batch_dataset.copy(),
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
        expected_df = self.rest_of_dataset[32820:].copy()
        fixValueInput = 0
        field = 'instrumentalness'
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset[32820:], fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(0), field=field)
        expected_df_copy = expected_df.copy()
        # Sustituir los valores 0 por los NaN en la columa field de expected_df
        expected_df_copy[field] = expected_df_copy[field].replace(fixValueInput, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        expected_df_copy[field] = expected_df_copy[field].interpolate(method='linear', limit_direction='both')

        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field]):
                # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                expected_df_copy.at[idx, field] = expected_df.at[idx, field]

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df_copy)
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset.copy(),
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
        result_df = self.invariants.checkInv_FixValue_NumOp(expected_df,
                                                            fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(3), axis_param=0)
        # Seleccionar solo las columnas numéricas del sub-DataFrame expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Iterar sobre cada columna numérica para encontrar y reemplazar el valor más cercano a fixValueInput
        for col in numeric_columns:
            # Calcular la diferencia absoluta con fixValueInput y excluir el propio fixValueInput
            diff = expected_df[col].apply(lambda x: abs(x - fixValueInput) if x != fixValueInput else np.inf)
            # Encontrar el índice del valor mínimo que no sea el propio fixValueInput
            idx_min = diff.idxmin()
            closest_value = expected_df.at[idx_min, col]
            # Sustituir el valor fixValueInput por el valor más cercano encontrado en la misma columna del sub-DataFrame expected_df
            expected_df[col] = expected_df[col].replace(fixValueInput, closest_value)

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
        result_df = self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset.copy(), fixValueInput=fixValueInput,
                                                            numOpOutput=Operation(0), axis_param=0)
        expected_df_copy = expected_df.copy()
        # Sustituir los valores 0 por los NaN en la columa field de expected_df
        expected_df_copy = expected_df_copy.replace(fixValueInput, np.nan)
        # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
        for col in expected_df_copy:
            if np.issubdtype(expected_df_copy[col].dtype, np.number):
                expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')

        for col in expected_df.columns:
            if np.issubdtype(expected_df[col].dtype, np.number):
                for idx in expected_df.index:
                    # Verificamos si el valor es NaN en el dataframe original
                    if pd.isnull(expected_df.at[idx, col]):
                        # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                        expected_df_copy.at[idx, col] = expected_df.at[idx, col]
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la excepción: se pasa axis_param a None cuando field=None y se lanza una excepción ValueError
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = 0.65
        expected_exception = ValueError
        # Verificar si el resultado obtenido coincide con el esperado
        with self.assertRaises(expected_exception):
            self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset.copy(),
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
            self.invariants.checkInv_FixValue_NumOp(self.rest_of_dataset.copy(),
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

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=65,
                                                            rightMargin=69, closureType=Closure(3),
                                                            dataTypeOutput=DataType(0), fixValueOutput='65<=Pop<=69',
                                                            field=field)

        expected_df['track_popularity'] = expected_df['track_popularity'].apply(
            lambda x: '65<=Pop<=69' if 65 <= x <= 69 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0.5,
                                                            rightMargin=1, closureType=Closure(0), fixValueOutput=2)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: 2 if (np.issubdtype(type(x), np.number) and
                                                                              ((x > 0.5) and (x < 1))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        field = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=65,
                                                                rightMargin=69, closureType=Closure(2),
                                                                fixValueOutput=101, field=field)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        field = 'speechiness'

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0.06,
                                                            rightMargin=0.1270, closureType=Closure(1),
                                                            fixValueOutput=33, field=field)

        expected_df['speechiness'] = expected_df['speechiness'].apply(lambda x: 33 if 0.06 < x <= 0.1270 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        field = 'p'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=65,
                                                                rightMargin=69, closureType=Closure(2),
                                                                fixValueOutput=101, field=field)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

    def execute_WholeDatasetTests_checkInv_Interval_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_FixValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field = 'track_popularity'

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=65,
                                                            rightMargin=69, closureType=Closure(2),
                                                            dataTypeOutput=DataType(0), fixValueOutput='65<=Pop<69',
                                                            field=field)

        expected_df['track_popularity'] = expected_df['track_popularity'].apply(
            lambda x: '65<=Pop<69' if 65 <= x < 69 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0.5,
                                                            rightMargin=1, closureType=Closure(0), fixValueOutput=2)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: 2 if (np.issubdtype(type(x), np.number) and
                                                                              ((x > 0.5) and (x < 1))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        field = 'track_name'
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=65,
                                                                rightMargin=69, closureType=Closure(2),
                                                                fixValueOutput=101, field=field)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        field = 'speechiness'

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0.06,
                                                            rightMargin=0.1270, closureType=Closure(1),
                                                            fixValueOutput=33, field=field)

        expected_df['speechiness'] = expected_df['speechiness'].apply(lambda x: 33 if 0.06 < x <= 0.1270 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

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

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(0), field=field)

        most_frequent_value = expected_df[field].mode().iloc[0]
        expected_df[field] = expected_df[field].apply(lambda x: most_frequent_value if 0.2 < x < 0.4 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.small_batch_dataset.copy()
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0.2,
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
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(1), axis_param=1)
        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i - 1] if np.issubdtype(
                                                                          type(value),
                                                                          np.number) and 0.2 < value < 0.4 and i > 0
                                                                      else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index), axis=1)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0.2,
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

        # Caso 5
        expected_df = self.small_batch_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(0), axis_param=None)

        most_frequent_value = expected_df.stack().value_counts().idxmax()
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: most_frequent_value if np.issubdtype(type(x), np.number)
                                                                   and 0.2 < x < 0.4 else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        expected_df = self.small_batch_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i + 1] if np.issubdtype(
                                                                          type(value), np.number) and 0.2 < value < 0.4
                                                                                                and i < len(
                                                                          row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                    leftMargin=0.2, rightMargin=0.4,
                                                                    closureType=Closure(0),
                                                                    derivedTypeOutput=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

    def execute_WholeDatasetTests_checkInv_Interval_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field = 'liveness'

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(0), field=field)

        most_frequent_value = expected_df[field].mode().iloc[0]
        expected_df[field] = expected_df[field].apply(lambda x: most_frequent_value if 0.2 < x < 0.4 else x)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        expected_df = self.rest_of_dataset.copy()
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0.2,
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
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(1), axis_param=1)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i - 1] if np.issubdtype(
                                                                          type(value),
                                                                          np.number) and 0.2 < value < 0.4 and i > 0
                                                                      else value for i, value in enumerate(row_or_col)],
                                                                     index=row_or_col.index), axis=1)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0.2,
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

        # Caso 5
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(0), axis_param=None)

        most_frequent_value = expected_df.stack().value_counts().idxmax()
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: most_frequent_value if np.issubdtype(type(x), np.number)
                                                                   and 0.2 < x < 0.4 else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                                                      row_or_col.iloc[i + 1] if np.issubdtype(
                                                                          type(value), np.number) and 0.2 < value < 0.4
                                                                                                and i < len(
                                                                          row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                    leftMargin=0.2, rightMargin=0.4,
                                                                    closureType=Closure(0),
                                                                    derivedTypeOutput=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

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

    def execute_SmallBatchTests_checkInv_Interval_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_Interval_NumOp
        """

        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=2, rightMargin=4,
                                                         closureType=Closure(1), numOpOutput=Operation(0), axis_param=0)
        expected_df_copy = expected_df.copy()

        for col in expected_df:
            if np.issubdtype(expected_df[col].dtype, np.number):
                expected_df_copy[col]=expected_df_copy[col].apply(lambda x: np.nan if (2<x<=4) else x)
                # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
                expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')
        # Iteramos sobre cada columna
        for col in expected_df.columns:
            # Para cada índice en la columna
            for idx in expected_df.index:
                # Verificamos si el valor es NaN en el dataframe original
                if pd.isnull(expected_df.at[idx, col]):
                    # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                    expected_df_copy.at[idx, col] = expected_df.at[idx, col]

        pd.testing.assert_frame_equal(result, expected_df_copy)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # # Caso 2
        # expected_df = self.small_batch_dataset.copy()
        # result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=2, rightMargin=4,
        #                                                  closureType=Closure(3), numOpOutput=Operation(0), axis_param=1)
        #
        # expected_df = expected_df.apply(
        #     lambda row: row.apply(lambda x: np.nan if np.issubdtype(type(x), np.number) and ((x >= 2) & (x <= 4)) else x).interpolate(
        #         method='linear', limit_direction='both'), axis=1)
        # pd.testing.assert_frame_equal(result, expected_df)
        # print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=2, rightMargin=4,
                                                             closureType=Closure(3), numOpOutput=Operation(0),
                                                             axis_param=None)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.small_batch_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=0, rightMargin=3,
                                                         closureType=Closure(0), numOpOutput=Operation(1),
                                                         axis_param=None)
        only_numbers_df = expected_df.select_dtypes(include=[np.number])
        # Calcular la media de estas columnas numéricas
        mean_value = only_numbers_df.mean().mean()
        # Reemplaza 'fixValueInput' con la media del DataFrame completo usando lambda
        expected_df = expected_df.apply(
            lambda col: col.apply(
                lambda x: mean_value if (np.issubdtype(type(x), np.number) and ((x > 0) & (x < 3))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        expected_df = self.small_batch_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=50, rightMargin=60,
                                                         closureType=Closure(0), numOpOutput=Operation(1), axis_param=0)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: col.mean() if (np.issubdtype(type(x), np.number)
                                            and ((x > 50) & (x < 60))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_df = self.small_batch_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=50, rightMargin=60,
                                                         closureType=Closure(2), numOpOutput=Operation(2), axis_param=None)
        only_numbers_df = expected_df.select_dtypes(include=[np.number])
        # Calcular la media de estas columnas numéricas
        median_value = only_numbers_df.median().median()
        # Reemplaza los valores en el intervalo con la mediana del DataFrame completo usando lambda
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: median_value if (np.issubdtype(type(x), np.number)
                                                    and ((x >= 50) & (x < 60))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 9
        expected_df = self.small_batch_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=23, rightMargin=25,
                                                         closureType=Closure(0), numOpOutput=Operation(3),
                                                         axis_param=None)

        #Sustituye los valores en el intervalo con el valor más cercano del dataframe completo
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: find_closest_value(expected_df.stack(), x)
            if np.issubdtype(type(x), np.number) and ((23<x) and (x<25)) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        expected_df = self.small_batch_dataset.copy()# Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=23, rightMargin=25,
                                                         closureType=Closure(0), numOpOutput=Operation(3), axis_param=0)

        # Reemplazar los valores en missing_values por el valor numérico más cercano a lo largo de las columnas y filas
        expected_df = expected_df.apply(lambda col: col.apply(lambda x:
                                find_closest_value(col, x) if np.issubdtype(type(x), np.number) and ((23<x) and (x<25)) else x),
                                                        axis=0)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        field = 'T'
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=2, rightMargin=4,
                                                             closureType=Closure(3), numOpOutput=Operation(0),
                                                             axis_param=None, field=field)
        print_and_log("Test Case 11 Passed: expected ValueError, got ValueError")

        # Caso 12
        expected_df = self.small_batch_dataset.copy()
        field = 'key'
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=2, rightMargin=4,
                                                         closureType=Closure(3), numOpOutput=Operation(0),
                                                         axis_param=None, field=field)
        expected_df[field] = expected_df[field].apply(lambda x: np.nan if (2 <= x <= 4) else x).interpolate(method='linear',
                                                                                                            limit_direction='both')
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        expected_df = self.small_batch_dataset.copy()
        field = 'key'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=2, rightMargin=4,
                                                         closureType=Closure(3), numOpOutput=Operation(1),
                                                         axis_param=None, field=field)
        expected_df[field] = expected_df[field].apply(lambda x: expected_df[field].mean() if (2 <= x <= 4) else x)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        expected_df = self.small_batch_dataset.copy()
        field = 'key'
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=2, rightMargin=4,
                                                         closureType=Closure(3), numOpOutput=Operation(2),
                                                         axis_param=None, field=field)
        median=expected_df[field].median()
        expected_df[field] = expected_df[field].apply(lambda x: median if (2 <= x <= 4) else x)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        expected_df = self.small_batch_dataset.copy()
        field = 'key'
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.small_batch_dataset.copy(), leftMargin=2,
                                                         rightMargin=4, closureType=Closure(2), numOpOutput=Operation(3),
                                                         axis_param=None, field=field)

        indice_row = []
        values = []
        processed = []
        closest_processed = []

        for index, value in expected_df[field].items():
            if 2<=value<4:
                indice_row.append(index)
                values.append(value)
        if values.__len__() > 0 and values is not None:
            processed.append(values[0])
            closest_processed.append(find_closest_value(expected_df[field], values[0]))
            for i in range(1, len(values)):
                if values[i] not in processed:
                    closest_value = find_closest_value(expected_df[field], values[i])
                    processed.append(values[i])
                    closest_processed.append(closest_value)
            for i, index in enumerate(indice_row):
                expected_df.at[index, field] = closest_processed[processed.index(values[i])]

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")


    def execute_WholeDatasetTests_checkInv_Interval_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_NumOp
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=2, rightMargin=4,
                                                         closureType=Closure(1), numOpOutput=Operation(0), axis_param=0)
        expected_df_copy = expected_df.copy()

        for col in expected_df:
            if np.issubdtype(expected_df[col].dtype, np.number):
                expected_df_copy[col]=expected_df_copy[col].apply(lambda x: np.nan if (2<x<=4) else x)
                # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
                expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')
        # Iteramos sobre cada columna
        for col in expected_df.columns:
            # Para cada índice en la columna
            for idx in expected_df.index:
                # Verificamos si el valor es NaN en el dataframe original
                if pd.isnull(expected_df.at[idx, col]):
                    # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                    expected_df_copy.at[idx, col] = expected_df.at[idx, col]

        pd.testing.assert_frame_equal(result, expected_df_copy)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # # Caso 2
        # expected_df = self.rest_of_dataset.copy()
        # result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=2, rightMargin=4,
        #                                                  closureType=Closure(3), numOpOutput=Operation(0), axis_param=1)
        #
        # expected_df = expected_df.apply(
        #     lambda row: row.apply(lambda x: np.nan if np.issubdtype(type(x), np.number) and ((x >= 2) & (x <= 4)) else x).interpolate(
        #         method='linear', limit_direction='both'), axis=1)
        # pd.testing.assert_frame_equal(result, expected_df)
        # print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=2, rightMargin=4,
                                                             closureType=Closure(3), numOpOutput=Operation(0),
                                                             axis_param=None)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=0, rightMargin=3,
                                                         closureType=Closure(0), numOpOutput=Operation(1),
                                                         axis_param=None)
        only_numbers_df = expected_df.select_dtypes(include=[np.number])
        # Calcular la media de estas columnas numéricas
        mean_value = only_numbers_df.mean().mean()
        # Reemplaza 'fixValueInput' con la media del DataFrame completo usando lambda
        expected_df = expected_df.apply(
            lambda col: col.apply(
                lambda x: mean_value if (np.issubdtype(type(x), np.number) and ((x > 0) & (x < 3))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        expected_df = self.rest_of_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=50, rightMargin=60,
                                                         closureType=Closure(0), numOpOutput=Operation(1), axis_param=0)

        expected_df = expected_df.apply(lambda col: col.apply(lambda x: col.mean() if (np.issubdtype(type(x), np.number)
                                            and ((x > 50) & (x < 60))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 7
        expected_df = self.rest_of_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=50, rightMargin=60,
                                                         closureType=Closure(2), numOpOutput=Operation(2), axis_param=None)
        only_numbers_df = expected_df.select_dtypes(include=[np.number])
        # Calcular la media de estas columnas numéricas
        median_value = only_numbers_df.median().median()
        # Reemplaza los valores en el intervalo con la mediana del DataFrame completo usando lambda
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: median_value if (np.issubdtype(type(x), np.number)
                                                    and ((x >= 50) & (x < 60))) else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 9
        expected_df = self.rest_of_dataset.copy()
        # start_time = time.time()

        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=23, rightMargin=25,
                                                         closureType=Closure(1), numOpOutput=Operation(3),
                                                         axis_param=None)

        only_numbers_df = expected_df.select_dtypes(include=[np.number])
        indice_row = []
        indice_col = []
        values = []
        for col in only_numbers_df.columns:
            for index, row in only_numbers_df.iterrows():
                if 23 < (row[col]) <= 25:
                    indice_row.append(index)
                    indice_col.append(col)
                    values.append(row[col])

        if values.__len__() > 0:
            processed = [values[0]]
            closest_processed = []
            closest_value = find_closest_value(only_numbers_df.stack(), values[0])
            closest_processed.append(closest_value)
            for i in range(len(values)):
                if values[i] not in processed:
                    closest_value = find_closest_value(only_numbers_df.stack(), values[i])
                    closest_processed.append(closest_value)
                    processed.append(values[i])

            # Recorrer todas las celdas del DataFrame
            for i in range(len(expected_df.index)):
                for j in range(len(expected_df.columns)):
                    # Obtener el valor de la celda actual
                    current_value = expected_df.iat[i, j]
                    # Verificar si el valor está en la lista de valores a reemplazar
                    if current_value in processed:
                        # Obtener el índice correspondiente en la lista de valores a reemplazar
                        replace_index = processed.index(current_value)
                        # Obtener el valor más cercano correspondiente
                        closest_value = closest_processed[replace_index]
                        # Reemplazar el valor en el DataFrame
                        expected_df.iat[i, j] = closest_value

        # end_time = time.time()
        #
        # # Calcula la diferencia de tiempo
        # execution_time = end_time - start_time
        #
        # print("Tiempo de ejecución:", execution_time, "segundos")

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")


        # Caso 10
        expected_df = self.rest_of_dataset.copy()
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=23,
                                                         rightMargin=25, closureType=Closure(0), numOpOutput=Operation(3),
                                                         axis_param=0)

        only_numbers_df = expected_df.select_dtypes(include=[np.number])
        for col in only_numbers_df.columns:
            indice_row = []
            indice_col = []
            values = []
            processed = []
            closest_processed = []

            for index, value in only_numbers_df[col].items():
                if (23 < value < 25):
                    indice_row.append(index)
                    indice_col.append(col)
                    values.append(value)

            if values.__len__() > 0 and values is not None:
                processed.append(values[0])
                closest_processed.append(find_closest_value(only_numbers_df[col], values[0]))

                for i in range(1, len(values)):
                    if values[i] not in processed:
                        closest_value = find_closest_value(only_numbers_df[col], values[i])
                        processed.append(values[i])
                        closest_processed.append(closest_value)

                for i, index in enumerate(indice_row):
                    expected_df.at[index, col] = closest_processed[processed.index(values[i])]

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        field = 'T'
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=2, rightMargin=4,
                                                             closureType=Closure(3), numOpOutput=Operation(0),
                                                             axis_param=None, field=field)
        print_and_log("Test Case 11 Passed: expected ValueError, got ValueError")

        # Caso 12
        expected_df = self.rest_of_dataset.copy()
        field = 'key'
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=2,
                                                         rightMargin=4, closureType=Closure(3), numOpOutput=Operation(0),
                                                         axis_param=None, field=field)
        expected_df_copy = expected_df.copy()

        if np.issubdtype(expected_df[field].dtype, np.number):
            expected_df_copy[field]=expected_df_copy[field].apply(lambda x: np.nan if (2<=x<=4) else x)
            # Definir el resultado esperado aplicando interpolacion lineal a nivel de columna
            expected_df_copy[field] = expected_df_copy[field].interpolate(method='linear', limit_direction='both')
        # Para cada índice en la columna
        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field]):
                # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                expected_df_copy.at[idx, field] = expected_df.at[idx, field]

        pd.testing.assert_frame_equal(result, expected_df_copy)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        expected_df = self.rest_of_dataset.copy()
        field = 'key'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=2, rightMargin=4,
                                                         closureType=Closure(3), numOpOutput=Operation(1),
                                                         axis_param=None, field=field)
        mean=expected_df[field].mean()
        expected_df[field] = expected_df[field].apply(lambda x: mean if (2 <= x <= 4) else x)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        expected_df = self.rest_of_dataset.copy()
        field = 'key'
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=2, rightMargin=4,
                                                         closureType=Closure(3), numOpOutput=Operation(2),
                                                         axis_param=None, field=field)
        median=expected_df[field].median()
        expected_df[field] = expected_df[field].apply(lambda x: median if (2 <= x <= 4) else x)
        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        expected_df = self.rest_of_dataset.copy()
        field = 'key'
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=self.rest_of_dataset.copy(), leftMargin=2,
                                                         rightMargin=4, closureType=Closure(2), numOpOutput=Operation(3),
                                                         axis_param=None, field=field)

        indice_row = []
        values = []
        processed = []
        closest_processed = []

        for index, value in expected_df[field].items():
            if 2<=value<4:
                indice_row.append(index)
                values.append(value)
        if values.__len__() > 0 and values is not None:
            processed.append(values[0])
            closest_processed.append(find_closest_value(expected_df[field], values[0]))
            for i in range(1, len(values)):
                if values[i] not in processed:
                    closest_value = find_closest_value(expected_df[field], values[i])
                    processed.append(values[i])
                    closest_processed.append(closest_value)
            for i, index in enumerate(indice_row):
                expected_df.at[index, field] = closest_processed[processed.index(values[i])]

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")









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

    def execute_SmallBatchTests_checkInv_SpecialValue_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_FixValue
        """
        # Caso 1 - Comprobar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de la columna
        # 'instrumentalness' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de prueba. Sobre un
        # dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el
        # resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [0, -1]
        fixValueOutput = "SpecialValue"
        field = 'instrumentalness'
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                                fixValueOutput=fixValueOutput, field=field)
        # Susituye los valores incluidos en missing_values, 0, None, así como los valores nulos de python de la cadena 'instrumentalness' por el valor de cadena 'SpecialValue' en expected_df
        expected_df['instrumentalness'] = expected_df['instrumentalness'].apply(lambda x: fixValueOutput if x in missing_values else x)

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2 - Comprobar la invariante: cambiar los valores missing 1 y 3, así como los valores nulos de python
        # de la columna 'key' por el valor de cadena 'SpecialValue' en el batch pequeño del dataset de
        # prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y
        # verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3]
        fixValueOutput = "SpecialValue"
        field = 'key'
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput, missing_values=missing_values,
                                                                fixValueOutput=fixValueOutput, field=field)
        expected_df['key'] = expected_df['key'].apply(lambda x: fixValueOutput if x in missing_values else x)

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3 - Comprobar la invariante cambiar los valores invalidos 1 y 3 por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 6]
        fixValueOutput = "SpecialValue"
        field = 'key'
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                missing_values=missing_values, fixValueOutput=fixValueOutput, field=field)
        expected_df['key'] = expected_df['key'].apply(lambda x: fixValueOutput if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar los outliers de la columna 'danceability' por el valor de cadena 'SpecialValue'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'danceability'
        fixValueOutput = "SpecialValue"
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                              fixValueOutput=fixValueOutput, field=field, axis_param=0)
        # Obtener los outliers de la columna 'danceability' y sustituirlos por el valor de cadena 'SpecialValue' en expected_df
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        if np.issubdtype(expected_df[field].dtype, np.number):
            expected_df[field] = expected_df[field].where(~((expected_df[field] < Q1 - 1.5 * IQR) | (expected_df[field] > Q3 + 1.5 * IQR)), other=fixValueOutput)
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 4 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log("Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar los valores outliers de todas las columnas del batch pequeño del dataset de
        # prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        fixValueOutput = "SpecialValue"
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                fixValueOutput=fixValueOutput, axis_param=0)

        # Obtener los outliers de todas las columnas que sean numéricas y sustituir los valores outliers por el valor de cadena 'SpecialValue' en expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            expected_df[col] = expected_df[col].where(~((expected_df[col] < Q1 - 1.5 * IQR) | (expected_df[col] > Q3 + 1.5 * IQR)), other=fixValueOutput)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar los valores invalid 1 y 3 en todas las columnas numéricas del batch pequeño
        # del dataset de prueba por el valor de cadena 'SpecialValue'. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3]
        fixValueOutput = 101
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                     missing_values=missing_values, fixValueOutput=fixValueOutput, axis_param=0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: cambiar los valores missing, así como el valor 0 y el -1 de todas las columnas numéricas
        # del batch pequeño del dataset de prueba por el valor 200. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [0, -1]
        fixValueOutput = 200
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                     missing_values=missing_values, fixValueOutput=fixValueOutput, axis_param=0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: cambiar los valores missing, así como el valor "Maroon 5" y "Katy Perry" de las columans de tipo string
        # del batch pequeño del dataset de prueba por el valor "SpecialValue". Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = ["Maroon 5", "Katy Perry"]
        fixValueOutput = "SpecialValue"
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                     missing_values=missing_values, fixValueOutput=fixValueOutput, axis_param=0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'p'
        fixValueOutput = "SpecialValue"
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                    specialTypeInput=specialTypeInput,
                                                                    fixValueOutput=fixValueOutput, field=field)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")


    def execute_WholeDatasetTests_checkInv_SpecialValue_FixValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_FixValue
        """
        # Caso 1 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [0, -1]
        fixValueOutput = "SpecialValue"
        field = 'instrumentalness'
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                   missing_values=missing_values,
                                                                   fixValueOutput=fixValueOutput, field=field)
        expected_df['instrumentalness'] = expected_df['instrumentalness'].replace(np.NaN, 0)
        expected_df['instrumentalness'] = expected_df['instrumentalness'].apply(
            lambda x: fixValueOutput if x in missing_values else x)
        # Cambair el dtype de la columna 'instrumentalness' a object
        expected_df['instrumentalness'] = expected_df['instrumentalness'].astype('object')
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3]
        fixValueOutput = "SpecialValue"
        field = 'key'
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                   missing_values=missing_values,
                                                                   fixValueOutput=fixValueOutput, field=field)
        expected_df['key'] = expected_df['key'].replace(np.NaN, 1)
        expected_df['key'] = expected_df['key'].apply(
            lambda x: fixValueOutput if x in missing_values else x)

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 6]
        fixValueOutput = "SpecialValue"
        field = 'key'
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                   missing_values=missing_values,
                                                                   fixValueOutput=fixValueOutput, field=field)
        expected_df['key'] = expected_df['key'].apply(lambda x: fixValueOutput if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'danceability'
        fixValueOutput = "SpecialValue"
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                   fixValueOutput=fixValueOutput, field=field,
                                                                   axis_param=0)
        # Obtener los outliers de la columna 'danceability' y sustituirlos por el valor de cadena 'SpecialValue' en expected_df
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        if np.issubdtype(expected_df[field].dtype, np.number):
            expected_df[field] = expected_df[field].where(
                ~((expected_df[field] < Q1 - 1.5 * IQR) | (expected_df[field] > Q3 + 1.5 * IQR)), other=fixValueOutput)
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 4 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 5 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        fixValueOutput = "SpecialValue"
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                   fixValueOutput=fixValueOutput, axis_param=0)

        # Obtener los outliers de todas las columnas que sean numéricas y sustituir los valores outliers por el valor de cadena 'SpecialValue' en expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            expected_df[col] = expected_df[col].where(
                ~((expected_df[col] < Q1 - 1.5 * IQR) | (expected_df[col] > Q3 + 1.5 * IQR)), other=fixValueOutput)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3]
        fixValueOutput = 101
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                   missing_values=missing_values,
                                                                   fixValueOutput=fixValueOutput, axis_param=0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [0, -1]
        fixValueOutput = 200
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                   missing_values=missing_values,
                                                                   fixValueOutput=fixValueOutput, axis_param=0)
        expected_df = expected_df.replace(np.NaN, 0)
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8 - Whole Dataset
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = ["Maroon 5", "Katy Perry"]
        fixValueOutput = "SpecialValue"
        result_df = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                   specialTypeInput=specialTypeInput,
                                                                   missing_values=missing_values,
                                                                   fixValueOutput=fixValueOutput, axis_param=0)
        expected_df = expected_df.replace(np.NaN, "Katy Perry")
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'p'
        fixValueOutput = "SpecialValue"
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                    specialTypeInput=specialTypeInput,
                                                                    fixValueOutput=fixValueOutput, field=field)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

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

    def execute_SmallBatchTests_checkInv_SpecialValue_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_SpecialValue_DerivedValue
        """
        # Caso 1
        # Comprobar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field = 'acousticness'
        derivedTypeOutput = DerivedType(0)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                    specialTypeInput=specialTypeInput,
                                                                       field=field, missing_values=missing_values,
                                                                          derivedTypeOutput=derivedTypeOutput)
        expected_df[field] = expected_df[field].replace(np.NaN, 1)
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        expected_df[field] = expected_df[field].apply(lambda x: most_frequent_value if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derivedTypeOutput = DerivedType(0)
        field = 'acousticness'
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                          missing_values=missing_values, field=field,
                                                                       derivedTypeOutput=derivedTypeOutput)
        expected_df[field] = expected_df[field].apply(lambda x: most_frequent_value if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derivedTypeOutput = DerivedType(0)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                            missing_values=missing_values,
                                                                       derivedTypeOutput=derivedTypeOutput, axis_param=0)
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
            most_frequent_list = expected_df[col].value_counts().index.tolist()
            most_frequent_value = most_frequent_list[0]
            expected_df[col] = expected_df[col].apply(lambda x: most_frequent_value if x in missing_values else x)
            # Convertir el tipo de dato de la columna al tipo que presente result en la columna
            expected_df[col] = expected_df[col].astype(result_df[col].dtype)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(0)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                          specialTypeInput=specialTypeInput,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                            missing_values=missing_values, axis_param=None)
        # Obtener el valor más frecuente de entre todas las columnas numéricas del dataframe
        most_frequent_list = expected_df.stack().value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: most_frequent_value if x in missing_values else x))
        for col in expected_df.columns:
            # Asignar el tipo de cada columna del dataframe result_df a la columna correspondiente en expected_df
            expected_df[col] = expected_df[col].astype(result_df[col].dtype)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(1)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                          derivedTypeOutput=derivedTypeOutput,
                                                                       axis_param=0, missing_values=missing_values)
        for col in expected_df.columns:

            # Iterar sobre la columna en orden inverso, comenzando por el penúltimo índice
            for i in range(len(expected_df[col]) - 2, -1, -1):
                if expected_df[col].iat[i] in missing_values and i > 0:
                    expected_df[col].iat[i] = expected_df[col].iat[i - 1]

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar los valores faltantes 1, 3, 0.13 y 0.187, así como los valores nulos de python por el valor derivado 2 (Next) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(2)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                        specialTypeInput=specialTypeInput,
                                                                       missing_values=missing_values,
                                                                            derivedTypeOutput=derivedTypeOutput, axis_param=0)
        # Asegúrate de que NaN esté incluido en la lista de valores faltantes para la comprobación, si aún no está incluido.
        missing_values = missing_values + [np.nan] if np.nan not in missing_values else missing_values

        for col in expected_df.columns:
            for i in range(len(expected_df[col]) - 1):  # Detenerse en el penúltimo elemento
                if expected_df[col].iat[i] in missing_values or pd.isnull(expected_df[col].iat[i]) and i < len(expected_df[col]) - 1:
                    expected_df[col].iat[i] = expected_df[col].iat[i + 1]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7 - Comprobar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1
        # (Previous) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(1)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                            missing_values=missing_values, axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8 - Comprobar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                            missing_values=missing_values, axis_param=None)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Comprobar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'p'
        derivedTypeOutput = DerivedType(0)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                            field=field)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Comprobar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'danceability'
        derivedTypeOutput = DerivedType(0)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                          derivedTypeOutput=derivedTypeOutput,
                                                                       field=field, axis_param=0)
        # Obtener el valor más frecuentemente repetido en la columna 'danceability'
        most_frequent_value = expected_df[field].mode()[0]
        # Obtener los outliers de la columna 'danceability' y sustituirlos por el valor más frecuente en expected_df
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        if np.issubdtype(expected_df[field].dtype, np.number):
            expected_df[field] = expected_df[field].where(
            ~((expected_df[field] < Q1 - 1.5 * IQR) | (expected_df[field] > Q3 + 1.5 * IQR)), other=most_frequent_value)
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 10 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")


        # Caso 11
        # Comprobar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'danceability'
        derivedTypeOutput = DerivedType(1)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                        specialTypeInput=specialTypeInput,
                                                                       field=field, derivedTypeOutput=derivedTypeOutput,
                                                                            axis_param=0)
        # En primer lugar, obtenemos los valores atípicos de la columna 'danceability'
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        if np.issubdtype(expected_df[field].dtype, np.number):
            # Sustituir los valores atípicos por el valor anterior en expected_df. Si el primer valor es un valor atípico, no se puede sustituir por el valor anterior.
            for i in range(1, len(expected_df[field])):
                expected_df[field].iat[i] = expected_df[field].iat[i - 1] if i>0 and ((expected_df[field].iat[i] < Q1 - 1.5 * IQR) or (expected_df[field].iat[i] > Q3 + 1.5 * IQR)) else expected_df[field].iat[i]
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 11 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 12
        # Comprobar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        derivedTypeOutput = DerivedType(2)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.small_batch_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                          derivedTypeOutput=derivedTypeOutput, axis_param=0)
        # Se obtienen los outliers de todas las columnas que sean numéricas y se sustituyen los valores outliers por el valor siguiente en expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns: # Sustituir los valores outliers por el valor siguiente
            # Obtiene los outliers de cada columna
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            for i in range(len(expected_df[col]) - 1):  # Detenerse en el penúltimo elemento
                if i < len(expected_df[col]) - 1 and expected_df[col].iat[i] < Q1 - 1.5 * IQR or expected_df[col].iat[i] > Q3 + 1.5 * IQR:
                    expected_df[col].iat[i] = expected_df[col].iat[i + 1]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")


    def execute_WholeDatasetTests_checkInv_SpecialValue_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_DerivedValue
        """
        # Caso 1
        # Comprobar la invariante: cambiar la lista de valores missing 1, 3 y 4 por el valor valor derivado 0 (most frequent value) en la columna 'acousticness'
        # en el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 0.146, 0.13, 0.187]
        field = 'acousticness'
        derivedTypeOutput = DerivedType(0)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       field=field, missing_values=missing_values,
                                                                       derivedTypeOutput=derivedTypeOutput)
        expected_df[field] = expected_df[field].replace(np.NaN, 1)
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        expected_df[field] = expected_df[field].apply(lambda x: most_frequent_value if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) de la columna 'acousticness'
        # en el batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derivedTypeOutput = DerivedType(0)
        field = 'acousticness'
        # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
        most_frequent_list = expected_df[field].value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       missing_values=missing_values, field=field,
                                                                       derivedTypeOutput=derivedTypeOutput)
        expected_df[field] = expected_df[field].apply(lambda x: most_frequent_value if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent) en
        # el dataframe completo del batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del dataset de prueba
        # cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [0.146, 0.13, 0.187]
        derivedTypeOutput = DerivedType(0)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       missing_values=missing_values,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                       axis_param=0)
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Obtener el valor más frecuente hasta ahora sin ordenarlos de menor a mayor. Quiero que sea el primer más frecuente que encuentre
            most_frequent_list = expected_df[col].value_counts().index.tolist()
            most_frequent_value = most_frequent_list[0]
            expected_df[col] = expected_df[col].apply(lambda x: most_frequent_value if x in missing_values else x)
            # Convertir el tipo de dato de la columna al tipo que presente result en la columna
            expected_df[col] = expected_df[col].astype(result_df[col].dtype)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 0 (Most Frequent) en todas
        # las columnas numéricas del batch grande del dataset de prueba. Sobre un dataframe de copia del batch grande del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(0)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                       missing_values=missing_values, axis_param=None)
        # Obtener el valor más frecuente de entre todas las columnas numéricas del dataframe
        most_frequent_list = expected_df.stack().value_counts().index.tolist()
        most_frequent_value = most_frequent_list[0]
        expected_df = expected_df.apply(
            lambda col: col.apply(lambda x: most_frequent_value if x in missing_values else x))
        for col in expected_df.columns:
            # Asignar el tipo de cada columna del dataframe result_df a la columna correspondiente en expected_df
            expected_df[col] = expected_df[col].astype(result_df[col].dtype)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1 (Previous) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(1)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                       axis_param=0, missing_values=missing_values)
        for col in expected_df.columns:

            # Iterar sobre la columna en orden inverso, comenzando por el penúltimo índice
            for i in range(len(expected_df[col]) - 2, -1, -1):
                if expected_df[col].iat[i] in missing_values and i>0:
                    expected_df[col].iat[i] = expected_df[col].iat[i - 1]

        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar los valores faltantes 1, 3, 0.13 y 0.187, así como los valores nulos de python por el valor derivado 2 (Next) en todas
        # las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(2)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       missing_values=missing_values,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                       axis_param=0)
        # Asegúrate de que NaN esté incluido en la lista de valores faltantes para la comprobación, si aún no está incluido.
        missing_values = missing_values + [np.nan] if np.nan not in missing_values else missing_values

        for col in expected_df.columns:
            for i in range(len(expected_df[col]) - 1):  # Detenerse en el penúltimo elemento
                if expected_df[col].iat[i] in missing_values or pd.isnull(expected_df[col].iat[i]) and i < len(expected_df[col]) - 1:
                    expected_df[col].iat[i] = expected_df[col].iat[i + 1]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7 - Comprobar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 1
        # (Previous) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(1)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                        specialTypeInput=specialTypeInput,
                                                                        derivedTypeOutput=derivedTypeOutput,
                                                                        missing_values=missing_values, axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8 - Comprobar la invariante: cambiar los valores invalidos 1, 3, 0.13 y 0.187 por el valor derivado 2
        # (Next) en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con
        # el esperado. Expected ValueError
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        derivedTypeOutput = DerivedType(2)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                        specialTypeInput=specialTypeInput,
                                                                        derivedTypeOutput=derivedTypeOutput,
                                                                        missing_values=missing_values, axis_param=None)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Comprobar la invariante: cambiar los valores outliers de una columna que no existe. Expected ValueError
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'p'
        derivedTypeOutput = DerivedType(0)
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                        specialTypeInput=specialTypeInput,
                                                                        derivedTypeOutput=derivedTypeOutput,
                                                                        field=field)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Comprobar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 0 (Most Frequent) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'danceability'
        derivedTypeOutput = DerivedType(0)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                       field=field, axis_param=0)
        # Obtener el valor más frecuentemente repetido en la columna 'danceability'
        most_frequent_value = expected_df[field].mode()[0]
        # Obtener los outliers de la columna 'danceability' y sustituirlos por el valor más frecuente en expected_df
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        if np.issubdtype(expected_df[field].dtype, np.number):
            expected_df[field] = expected_df[field].where(
                ~((expected_df[field] < Q1 - 1.5 * IQR) | (expected_df[field] > Q3 + 1.5 * IQR)),
                other=most_frequent_value)
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 10 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 11
        # Comprobar la invariante: cambiar los valores outliers de una columna especifica por el valor derivado 1
        # (Previous) en el batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        field = 'danceability'
        derivedTypeOutput = DerivedType(1)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       field=field, derivedTypeOutput=derivedTypeOutput,
                                                                       axis_param=0)
        # En primer lugar, obtenemos los valores atípicos de la columna 'danceability'
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        if np.issubdtype(expected_df[field].dtype, np.number):
            # Sustituir los valores atípicos por el valor anterior en expected_df. Si el primer valor es un valor atípico, no se puede sustituir por el valor anterior.
            for i in range(1, len(expected_df[field])):
                expected_df[field].iat[i] = expected_df[field].iat[i - 1] if i > 0 and (
                            (expected_df[field].iat[i] < Q1 - 1.5 * IQR) or (
                                expected_df[field].iat[i] > Q3 + 1.5 * IQR)) else expected_df[field].iat[i]
            pd.testing.assert_frame_equal(result_df, expected_df)
            print_and_log("Test Case 11 Passed: the function returned the expected dataframe")
        else:
            # Print and log a warning message indicating the column is not numeric
            print_and_log(
                "Warning: The column 'danceability' is not numeric. The function returned the original dataframe")

        # Caso 12
        # Comprobar la invariante: cambiar los valores outliers del dataframe completo por el valor derivado 2 (Next)
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        derivedTypeOutput = DerivedType(2)
        result_df = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=self.rest_of_dataset.copy(),
                                                                       specialTypeInput=specialTypeInput,
                                                                       derivedTypeOutput=derivedTypeOutput,
                                                                       axis_param=0)
        # Se obtienen los outliers de todas las columnas que sean numéricas y se sustituyen los valores outliers por el valor siguiente en expected_df
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:  # Sustituir los valores outliers por el valor siguiente
            # Obtiene los outliers de cada columna
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            for i in range(len(expected_df[col]) - 1):  # Detenerse en el penúltimo elemento
                if i < len(expected_df[col]) - 1 and expected_df[col].iat[i] < Q1 - 1.5 * IQR or \
                        expected_df[col].iat[i] > Q3 + 1.5 * IQR:
                    expected_df[col].iat[i] = expected_df[col].iat[i + 1]
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

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
        #MISSING
        # Caso 1
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(0)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                               specialTypeInput=specialTypeInput,
                                                               numOpOutput=numOpOutput, missing_values=missing_values,
                                                               axis_param=0)
        # Aplica la interpolación lineal a los valores faltantes y valores nulos a través de todas las columnas del dataframe
        # En primer lugar, se reemplazan los valores nulos y los valores faltantes por NaN
        missing_values = missing_values + [np.nan] if np.nan not in missing_values else missing_values
        replaced_df = expected_df.replace(missing_values, np.nan)
        # Selecciona las columnas numéricas del dataframe
        numeric_columns = replaced_df.select_dtypes(include=np.number).columns
        # Aplica la interpolación lineal a través de todas las columnas numéricas del dataframe
        expected_df[numeric_columns] = replaced_df[numeric_columns].interpolate(method='linear', axis=0, limit_direction='both')
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(1)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # Obtener la media de las columnas numéricas del dataframe
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la media de todas las columnas numéricas
        mean_value = expected_df[numeric_columns].mean().mean()
        # Sustituir los valores faltantes y valores nulos por la media de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, mean_value)
        # Susituir los valores nulos por la media de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(np.nan, mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3 Se aplica la mediana de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(2)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].median()
        for col in numeric_columns:
            # Sustituir los valores faltantes y valores nulos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
            # Sustituir los valores nulos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(np.nan, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(3)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                               specialTypeInput=specialTypeInput,
                                                               numOpOutput=numOpOutput, missing_values=missing_values,
                                                               axis_param=0)
        # Para cada columna numérica, se sustituyen los valores faltantes y valores nulos por el valor más cercano
        # en el dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Obtener el valor más cercano a cada valor faltante y valor nulo en la columna
            expected_df[col] = expected_df[col].apply(lambda x: find_closest_value(expected_df[col].tolist(), x) if x in missing_values or pd.isnull(x) else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto. Se calcula el closest sobre el dataframe entero en relación a los valores faltantes y valores nulos.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(3)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # Sustituir los valores faltantes y valores nulos por el valor más cercano en el dataframe
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: find_closest_value(expected_df.stack().tolist(), x) if x in missing_values or pd.isnull(x) else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las
        # columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df =  self.small_batch_dataset.copy()
        expected_df_copy = expected_df.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(0)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary= self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Aplicar la interpolación lineal a los valores invalidos en todas las columnas del dataframe
        # En primer lugar, se reemplazan los valores invalidos por NaN
        replaced_df = expected_df.replace(missing_values, np.nan)
        # Selecciona las columnas numéricas del dataframe
        numeric_columns = replaced_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Se sustituyen los valores invalidos por NaN
            expected_df[col] = replaced_df[col].replace(missing_values, np.nan)
            # Aplica la interpolación lineal a través de todas las columnas numéricas del dataframe
            expected_df[col] = replaced_df[col].interpolate(method='linear', axis=0, limit_direction='both')
        # Se asignan los valores nan o null del dataframe 'expected_df_copy' al dataframe 'expected_df'
        for col in numeric_columns:
            for idx, row in expected_df_copy.iterrows():
                if pd.isnull(row[col]):
                    expected_df.at[idx, col] = np.nan
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(1)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # Obtener la media de las columnas numéricas del dataframe
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la media de todas las columnas numéricas
        mean_value = expected_df[numeric_columns].mean().mean()
        # Sustituir los valores invalidos por la media de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: aplicar la media de cada columna numérica a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(1)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].mean()
        for col in numeric_columns:
            # Sustituir los valores invalidos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la invariante: aplicar la mediana de cada columna numérica del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(2)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].median()
        for col in numeric_columns:
            # Sustituir los valores invalidos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Comprobar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(2)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de todas las columnas numéricas
        median_value = expected_df[numeric_columns].median().median()
        # Sustituir los valores invalidos por la mediana de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, median_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Comprobar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(3)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Sustituir los valores invalidos por el valor más cercano en el dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            expected_df[col] = expected_df[col].apply(lambda x: find_closest_value(expected_df[col].tolist(), x) if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 12
        # Comprobar la invariante: aplicar el closest a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(3)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # Sustituir los valores invalidos por el valor más cercano en el dataframe
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: find_closest_value(expected_df.stack().tolist(), x) if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        # Comprobar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        numOpOutput = Operation(0)
        field = 'danceability'
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, field=field, axis_param=0)
        expected_df_copy = expected_df.copy()
        # Aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # En primer lugar, se reemplazan los valores outliers por NaN
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        lowest_value=Q1 - 1.5 * IQR
        upper_value=Q3 + 1.5 * IQR
        for idx, value in expected_df[field].items():
            # Sustituir los valores outliers por NaN
            if expected_df.at[idx, field] < lowest_value or expected_df.at[idx, field] > upper_value:
                expected_df.at[idx, field] = np.NaN
        expected_df[field] = expected_df[field].interpolate(method='linear', limit_direction='both')
        # Para cada índice en la columna
        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field]):
                # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                expected_df_copy.at[idx, field] = expected_df.at[idx, field]

        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        # Comprobar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        numOpOutput = Operation(0)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, axis_param=0)
        expected_df_copy=expected_df.copy()
        # Aplicar la interpolación lineal a los valores outliers de cada columna del dataframe
        # En primer lugar, se reemplazan los valores outliers por NaN
        for col in expected_df.select_dtypes(include=np.number).columns:
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Para cada valor en la columna, bucle for
            for i in range(len(expected_df[col])):
                # Sustituir los valores outliers por NaN
                if expected_df[col].iat[i] < Q1 - 1.5 * IQR or expected_df[col].iat[i] > Q3 + 1.5 * IQR:
                    expected_df_copy[col].iat[i] = np.NaN
                # Aplica la interpolación lineal a través de la columna en cuestión del dataframe
            expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')
            for col in expected_df.columns:
                # Para cada índice en la columna
                for idx in expected_df.index:
                    # Verificamos si el valor es NaN en el dataframe original
                    if pd.isnull(expected_df.at[idx, col]):
                        # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                        expected_df_copy.at[idx, col] = expected_df.at[idx, col]

        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        # Comprobar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        numOpOutput = Operation(1)
        field = 'danceability'
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, field=field, axis_param=0)
        # Obtener la media de la columna 'danceability'
        mean_value = expected_df[field].mean()
        # Obtener los outliers de la columna 'danceability'
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        # Sustituir los valores outliers por la media de la columna 'danceability'
        expected_df[field] = expected_df[field].where(~((expected_df[field] < Q1 - 1.5 * IQR) | (expected_df[field] > Q3 + 1.5 * IQR)), other=mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

        # Caso 16
        # Comprobar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.small_batch_dataset.copy()
        specialTypeInput = SpecialType(2)
        numOpOutput = Operation(1)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.small_batch_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, axis_param=0)
        for col in expected_df.select_dtypes(include=np.number).columns:
            # Obtener los outliers de la columna
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            for idx in range(len(expected_df[col])):
                # Obtener la media de la columna
                mean_value = expected_df[col].mean()
                expected_df[col].iat[idx] = expected_df[col].iat[idx] if not (expected_df[col].iat[idx] < Q1 - 1.5 * IQR or expected_df[col].iat[idx] > Q3 + 1.5 * IQR) else mean_value
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 16 Passed: the function returned the expected dataframe")


        # Caso 11
        # Comprobar la invariante: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las


        # # Caso 5
        # # Probamos a aplicar la operación closest sobre un dataframe correcto
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        # missing_values = [3, 4]
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(0),
        #                                                         numOpOutput=Operation(3), missing_values=missing_values,
        #                                                         axis_param=0)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 5 Passed: got the dataframe expected")
        #
        # # Invalid
        # # Caso 6
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        # missing_values = [1, 3, 4]
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 2, 2.0, 2], 'B': [2, 2 + 4 / 3, 2 + 8 / 3, 6, 12], 'C': [10, 7.5, 5, 2.5, 0],
        #      'D': [8.2, 8.2, 6, 4, 2]})
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(1),
        #                                                         numOpOutput=Operation(0), missing_values=missing_values,
        #                                                         axis_param=0)
        # result_df['A'] = result_df['A'].astype('float64')
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 6 Passed: got the dataframe expected")
        #
        # # Caso 7
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        # missing_values = [3, 4]
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 3.61, 3.61, 1], 'B': [2, 3.61, 3.61, 6, 12], 'C': [10, 1, 3.61, 3.61, 0],
        #      'D': [1, 8.2, 6, 1, 2]})
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(1),
        #                                                         numOpOutput=Operation(1), missing_values=missing_values,
        #                                                         axis_param=None)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 7 Passed: got the dataframe expected")
        #
        # # Caso 8
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        # missing_values = [1, 4]
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 3, 3.5, 1.5], 'B': [2, 3, 3.5, 6, 12], 'C': [10, 2.5, 3, 3, 0], 'D': [1.5, 8.2, 6, 3.5, 2]})
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(1),
        #                                                         numOpOutput=Operation(2), missing_values=missing_values,
        #                                                         axis_param=1)
        # # Cambiar el tipo de la columna 'A' a float64
        # result_df['A'] = result_df['A'].astype('float64')
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 8 Passed: got the dataframe expected")
        #
        # # Caso 9
        # # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        # missing_values = [1, 3, 4]
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 2, 3, 0], 'B': [2, 2, 3, 6, 12], 'C': [10, 3, 4, np.NaN, 0], 'D': [2, 8.2, 2, 2, 2]})
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(1),
        #                                                         numOpOutput=Operation(3), missing_values=missing_values,
        #                                                         axis_param=0)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 9 Passed: got the dataframe expected")
        #
        # # Caso 10
        # # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        # missing_values = [3, 4]
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})
        #
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(1),
        #                                                         numOpOutput=Operation(3), missing_values=missing_values,
        #                                                         axis_param=0)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 10 Passed: got the dataframe expected")
        #
        # # Outliers
        # # Caso 11
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [1, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        # expected_df = expected_df.astype({
        #     'D': 'float64',  # Convertir D a float64
        #     'B': 'float64',  # Convertir B a float64
        #     'C': 'float64'  # Convertir C a float64
        # })
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(2),
        #                                                         numOpOutput=Operation(0), axis_param=0)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 11 Passed: got the dataframe expected")
        #
        # # Caso 12
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        # datadic = datadic.astype({
        #     'B': 'float64',  # Convertir B a float64
        #     'C': 'float64'  # Convertir C a float64
        # })
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 3.61], 'C': [3.61, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        #
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(2),
        #                                                         numOpOutput=Operation(1), missing_values=None,
        #                                                         axis_param=None)
        #
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 12 Passed: got the dataframe expected")
        #
        # # Caso 13
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        # datadic = datadic.astype({
        #     'B': 'float64',  # Convertir B a float64
        #     'C': 'float64'  # Convertir B a float64
        # })
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 1.5], 'C': [1.5, 1, 3, 3, 0], 'D': [1, 2.5, 6, 1, 2]})
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(2),
        #                                                         numOpOutput=Operation(2), missing_values=None,
        #                                                         axis_param=1)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 13 Passed: got the dataframe expected")
        #
        # # Caso 14
        # # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 3, 3, 1, 2]})
        # expected_df = expected_df.astype({
        #     'D': 'float64'  # Convertir D a float64
        # })
        #
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(2),
        #                                                         numOpOutput=Operation(3), missing_values=None,
        #                                                         axis_param=0)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 14 Passed: got the dataframe expected")
        #
        # # Caso 15
        # # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [10, 6, 3, 3, 0], 'D': [1, 2, 2, 1, 2]})
        # expected_df = expected_df.astype({
        #     'D': 'float64'  # Convertir D a float64
        # })
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(2),
        #                                                         numOpOutput=Operation(3), axis_param=0)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 15 Passed: got the dataframe expected")
        #
        # # Caso 16
        # # Probamos a aplicar la operación mean sobre un field concreto
        # datadic = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        # expected_df = pd.DataFrame(
        #     {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 2.84, 2, 1, 2]})
        # expected_df = expected_df.astype({
        #     'D': 'float64'  # Convertir D a float64
        # })
        # field = 'D'
        # missing_values = [8.2]
        # result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic.copy(),
        #                                                         specialTypeInput=SpecialType(2),
        #                                                         numOpOutput=Operation(1), missing_values=missing_values,
        #                                                         axis_param=0, field=field)
        # pd.testing.assert_frame_equal(expected_df, result_df)
        # print_and_log("Test Case 16 Passed: got the dataframe expected")

    # TODO: Implement the invariant tests with external dataset
    def execute_WholeDatasetTests_checkInv_SpecialValue_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_SpecialValue_NumOp
        """
        # MISSING
        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(0)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Aplica la interpolación lineal a los valores faltantes y valores nulos a través de todas las columnas del dataframe
        # En primer lugar, se reemplazan los valores nulos y los valores faltantes por NaN
        missing_values = missing_values + [np.nan] if np.nan not in missing_values else missing_values
        replaced_df = expected_df.replace(missing_values, np.nan)
        #Selecciona las columnas numéricas del dataframe
        numeric_columns = replaced_df.select_dtypes(include=np.number).columns
        # Aplica la interpolación lineal a través de todas las columnas numéricas del dataframe
        expected_df[numeric_columns] = replaced_df[numeric_columns].interpolate(method='linear', axis=0, limit_direction='both')
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2 - Se aplica la media de todas las columnas numéricas del dataframe a los valores faltantes y valores nulos de python
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(1)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # Obtener la media de las columnas numéricas del dataframe
        # Obtener las columnas numéricas
        only_numbers_df = expected_df.select_dtypes(include=[np.number])
        # Obtener la media de todas las columnas numéricas
        mean_value = only_numbers_df.mean().mean()
        # Sustituir los valores faltantes y valores nulos por la media de todas las columnas numéricas
        expected_df = expected_df.apply(
                lambda col: col.apply(lambda x: mean_value if (x in missing_values or pd.isnull(x)) else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3 Se aplica la mediana de todas las columnas numéricas del dataframe a los valores faltantes y valores
        # nulos de python
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(2)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].median()
        for col in numeric_columns:
            # Sustituir los valores faltantes y valores nulos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
            # Sustituir los valores nulos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(np.nan, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos) y sobre
        # cada columna del dataframe.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(3)
        # Al ser una operación de missing a closest y existen valores nulos, se devolverá un ValueError ya que
        # no se puede calcular el valor más cercano a un valor nulo
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                 specialTypeInput=specialTypeInput,
                                                                 numOpOutput=numOpOutput,
                                                                 missing_values=missing_values,
                                                                 axis_param=0)
        print_and_log("Test Case 4 Passed: Expected ValueError, got ValueError")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto. Se calcula el closest sobre el dataframe entero en relación a los valores faltantes y valores nulos.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(0)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(3)
        # Al ser una operación de missing a closest y no existen valores nulos, se devolverá un ValueError ya que
        # no se puede calcular el valor más cercano a un valor nulo
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                 specialTypeInput=specialTypeInput,
                                                                 numOpOutput=numOpOutput,
                                                                 missing_values=missing_values,
                                                                 axis_param=None)
        print_and_log("Test Case 5 Passed: Expected ValueError, got ValueError")

        # Caso 6
        # Comprobar la invariante: aplicar la interpolación lineal a los valores invalidos 1, 3, 0.13 y 0.187 en todas las
        # columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del dataset de
        # prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        expected_df_copy = expected_df.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(0)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Aplicar la interpolación lineal a los valores invalidos en todas las columnas del dataframe
        # En primer lugar, se reemplazan los valores invalidos por NaN
        replaced_df = expected_df.replace(missing_values, np.nan)
        # Selecciona las columnas numéricas del dataframe
        numeric_columns = replaced_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            # Se sustituyen los valores invalidos por NaN
            expected_df[col] = replaced_df[col].replace(missing_values, np.nan)
            # Aplica la interpolación lineal a través de todas las columnas numéricas del dataframe
            expected_df[col] = replaced_df[col].interpolate(method='linear', axis=0, limit_direction='both')
        # Se asignan los valores nan o null del dataframe 'expected_df_copy' al dataframe 'expected_df'
        for col in numeric_columns:
            for idx, row in expected_df_copy.iterrows():
                if pd.isnull(row[col]):
                    expected_df.at[idx, col] = np.nan
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(1)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # Obtener la media de las columnas numéricas del dataframe
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la media de todas las columnas numéricas
        mean_value = expected_df[numeric_columns].mean().mean()
        # Sustituir los valores invalidos por la media de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: aplicar la media de cada columna numérica a los valores invalidos 1, 3, 0.13 y 0.187
        # en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño
        # del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(1)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].mean()
        for col in numeric_columns:
            # Sustituir los valores invalidos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la invariante: aplicar la mediana de cada columna numérica del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(2)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de cada columna numérica
        median_values = expected_df[numeric_columns].median()
        for col in numeric_columns:
            # Sustituir los valores invalidos por la mediana de cada columna numérica
            expected_df[col] = expected_df[col].replace(missing_values, median_values[col])
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Comprobar la invariante: aplicar la mediana de todas las columnas numéricas del dataframe a los valores invalidos
        # 1, 3, 0.13 y 0.187 en todas las columnas del batch pequeño del dataset de prueba. Sobre un dataframe de copia
        # del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido
        # coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [1, 3, 0.13, 0.187]
        numOpOutput = Operation(2)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # Obtener las columnas numéricas
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        # Obtener la mediana de todas las columnas numéricas
        median_value = expected_df[numeric_columns].median().median()
        # Sustituir los valores invalidos por la mediana de todas las columnas numéricas
        expected_df[numeric_columns] = expected_df[numeric_columns].replace(missing_values, median_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Comprobar la invariante: aplicar el closest al valor invalido 0.13
        # en cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [0.13]
        numOpOutput = Operation(3)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=0)
        # Sustituir los valores invalidos por el valor más cercano en el dataframe
        numeric_columns = expected_df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            expected_df[col] = expected_df[col].apply(
                lambda x: find_closest_value(expected_df[col].tolist(), x) if x in missing_values else x)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 11.1
        # Comprobar la invariante: aplicar el closest al valor invalido 0.13
        # en todas las columnas del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente
        # y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(1)
        missing_values = [0.13]
        numOpOutput = Operation(3)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, missing_values=missing_values,
                                                                axis_param=None)
        # # Sustituir los valores invalidos por el valor más cercano en el dataframe
        expected_df = expected_df.apply(lambda col: col.apply(
            lambda x: find_closest_value(expected_df.stack().tolist(), x) if x in missing_values else x))
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 11.1 Passed: the function returned the expected dataframe")

        # Caso 13
        # Comprobar la invariante: aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        numOpOutput = Operation(0)
        field = 'danceability'
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, field=field, axis_param=0)
        expected_df_copy = expected_df.copy()
        # Aplicar la interpolación lineal a los valores outliers de la columna 'danceability'
        # En primer lugar, se reemplazan los valores outliers por NaN
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        lowest_value=Q1 - 1.5 * IQR
        upper_value=Q3 + 1.5 * IQR
        for idx, value in expected_df[field].items():
            # Sustituir los valores outliers por NaN
            if expected_df.at[idx, field] < lowest_value or expected_df.at[idx, field] > upper_value:
                expected_df.at[idx, field] = np.NaN
        expected_df[field] = expected_df[field].interpolate(method='linear', limit_direction='both')
        # Para cada índice en la columna
        for idx in expected_df.index:
            # Verificamos si el valor es NaN en el dataframe original
            if pd.isnull(expected_df.at[idx, field]):
                # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                expected_df_copy.at[idx, field] = expected_df.at[idx, field]

        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        # Comprobar la invariante: aplicar la interpolación lineal a los valores outliers de cada columna del batch pequeño del dataset de prueba.
        # Sobre un dataframe de copia del batch pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        numOpOutput = Operation(0)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, axis_param=0)
        expected_df_copy=expected_df.copy()
        # Aplicar la interpolación lineal a los valores outliers de cada columna del dataframe
        # En primer lugar, se reemplazan los valores outliers por NaN
        for col in expected_df.select_dtypes(include=np.number).columns:
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Para cada valor en la columna, bucle for
            for i in range(len(expected_df[col])):
                # Sustituir los valores outliers por NaN
                if expected_df[col].iat[i] < Q1 - 1.5 * IQR or expected_df[col].iat[i] > Q3 + 1.5 * IQR:
                    expected_df_copy[col].iat[i] = np.NaN
                # Aplica la interpolación lineal a través de la columna en cuestión del dataframe
            expected_df_copy[col] = expected_df_copy[col].interpolate(method='linear', limit_direction='both')
            for col in expected_df.columns:
                # Para cada índice en la columna
                for idx in expected_df.index:
                    # Verificamos si el valor es NaN en el dataframe original
                    if pd.isnull(expected_df.at[idx, col]):
                        # Reemplazamos el valor con el correspondiente de dataDictionary_copy_copy
                        expected_df_copy.at[idx, col] = expected_df.at[idx, col]

        pd.testing.assert_frame_equal(result_df, expected_df_copy)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        # Comprobar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de la columna 'danceability' del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch
        # pequeño del dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        numOpOutput = Operation(1)
        field = 'danceability'
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, field=field, axis_param=0)
        # Obtener la media de la columna 'danceability'
        mean_value = expected_df[field].mean()
        print("Media calculada en los tests: ", mean_value)
        # Obtener los outliers de la columna 'danceability'
        Q1 = expected_df[field].quantile(0.25)
        Q3 = expected_df[field].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_col = Q1 - 1.5 * IQR
        upper_bound_col = Q3 + 1.5 * IQR
        # Sustituir los valores outliers por la media de la columna 'danceability'
        expected_df[field] = expected_df[field].where(
            ~((expected_df[field] < lower_bound_col) | (expected_df[field] > upper_bound_col)), other=mean_value)
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

        # Caso 16
        # Comprobar la invariante: aplicar la media de todas las columnas numéricas del dataframe a los valores outliers
        # de cada columna del batch pequeño del dataset de prueba. Sobre un dataframe de copia del batch pequeño del
        # dataset de prueba cambiar los valores manualmente y verificar si el resultado obtenido coincide con el esperado.
        expected_df = self.rest_of_dataset.copy()
        specialTypeInput = SpecialType(2)
        numOpOutput = Operation(1)
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=self.rest_of_dataset.copy(),
                                                                specialTypeInput=specialTypeInput,
                                                                numOpOutput=numOpOutput, axis_param=0)
        for col in expected_df.select_dtypes(include=np.number).columns:
            # Obtener los outliers de la columna
            Q1 = expected_df[col].quantile(0.25)
            Q3 = expected_df[col].quantile(0.75)
            IQR = Q3 - Q1
            for idx in range(len(expected_df[col])):
                # Obtener la media de la columna
                mean_value = expected_df[col].mean()
                expected_df[col].iat[idx] = expected_df[col].iat[idx] if not (
                            expected_df[col].iat[idx] < Q1 - 1.5 * IQR or expected_df[col].iat[
                        idx] > Q3 + 1.5 * IQR) else mean_value
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 16 Passed: the function returned the expected dataframe")
