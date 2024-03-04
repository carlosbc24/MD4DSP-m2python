import os
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

from functions.contract_invariants import ContractsInvariants
from helpers.enumerations import DerivedType, Closure, DataType
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
        fixValueInput = pd.to_datetime('2019-07-05')
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

        # Caso 6
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
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
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
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: cambiar el valor fijo de tipo TIME '2019-07-05' de la columna track_album_release_date
        # por el valor fijo de tipo boolean True sobre el batch grande del dataset de prueba. Sobre un dataframe de
        # copia del batch grande del dataset de prueba cambiar los valores manualmente y verificar si el resultado
        # obtenido coincide con el esperado
        expected_df = self.rest_of_dataset.copy()
        fixValueInput = pd.to_datetime('2019-07-05')
        fixValueOutput = True
        result_df = self.invariants.checkInv_FixValue_FixValue(self.rest_of_dataset,
                                                               fixValueInput=fixValueInput,
                                                               fixValueOutput=fixValueOutput)
        expected_df['track_album_release_date'] = expected_df['track_album_release_date'].replace(fixValueInput,
                                                                                                  fixValueOutput)
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
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
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
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
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
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
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

    # TODO: Implement the invariant tests with external dataset
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

    # TODO: Implement the invariant tests with external dataset
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
        for idx in katy_perry_indices:
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

        # Caso 3
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor derivado 3 (Next) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic,
                                                                fixValueInput=0,
                                                                derivedTypeOutput=DerivedType(2), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor más frecuente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic,
                                                                fixValueInput=5,
                                                                derivedTypeOutput=DerivedType(0), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 3], 'B': [1, 8, 4, 4, 4], 'C': [1, 2, 3, 4, 3]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor más frecuente a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 7, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic,
                                                                fixValueInput=5,
                                                                derivedTypeOutput=DerivedType(0), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 8], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 7, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor previo a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic,
                                                                fixValueInput=5,
                                                                derivedTypeOutput=DerivedType(1), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 5], 'B': [1, 8, 3, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 4, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic,
                                                                fixValueInput=5,
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 6, 6, 8, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, "Ainhoa", "Ainhoa", 5], 'B': [1, 8, "Ainhoa", 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"],
             'D': [4, 5, 6, 5, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic,
                                                                fixValueInput="Ainhoa",
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, "Ainhoa", 5, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"], 'D': [4, 5, 6, 5, 8]})
        expected = expected.astype({
            'A': 'object',  # Convertir A a object
            'B': 'object',  # Convertir B a int64
            'C': 'object',  # Convertir C a object
            'D': 'int64'  # Convertir D a object
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Comprobar la invariante: cambiar el valor fijo "Ana" por el valor más frecuente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", "Ana", pd.to_datetime('2021-01-01')],
                                'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')],
                                'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"],
                                'D': [pd.to_datetime('2021-01-01'), 5, "Ana", 5, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic,
                                                                fixValueInput="Ana",
                                                                derivedTypeOutput=DerivedType(0), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", pd.to_datetime('2021-01-01'),
                                       pd.to_datetime('2021-01-01')],
                                 'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')],
                                 'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"],
                                 'D': [pd.to_datetime('2021-01-01'), 5, 5, 5, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

    # TODO: Implement the invariant tests with external dataset
    def execute_WholeDatasetTests_checkInv_FixValue_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_DerivedValue
        """

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

    # TODO: Implement the invariant tests with external dataset
    def execute_SmallBatchTests_checkInv_FixValue_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using a small batch of the dataset for the function checkInv_FixValue_NumOp
        """

    # TODO: Implement the invariant tests with external dataset
    def execute_WholeDatasetTests_checkInv_FixValue_NumOp_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_FixValue_NumOp
        """

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
        field='track_popularity'

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.small_batch_dataset, leftMargin=65,
                                                            rightMargin=69, closureType=Closure(3),
                                                            dataTypeOutput=DataType(0), fixValueOutput='65<=Pop<=69',
                                                            field=field)

        expected_df['track_popularity'] = expected_df['track_popularity'].apply(lambda x: '65<=Pop<=69' if 65 <= x <= 69 else x)

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
        field='track_popularity'

        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=self.rest_of_dataset, leftMargin=65,
                                                            rightMargin=69, closureType=Closure(2),
                                                            dataTypeOutput=DataType(0), fixValueOutput='65<=Pop<69',
                                                            field=field)

        expected_df['track_popularity'] = expected_df['track_popularity'].apply(lambda x: '65<=Pop<69' if 65 <= x < 69 else x)

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
        field='liveness'

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
                                        row_or_col.iloc[i - 1] if np.issubdtype(type(value), np.number) and 0.2 < value < 0.4 and i > 0
                                        else value for i, value in enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.small_batch_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                        row_or_col.iloc[i + 1] if np.issubdtype(type(value), np.number) and 0.2 < value < 0.4
                                                and i < len(row_or_col) - 1 else value for i, value in
                                                                      enumerate(row_or_col)], index=row_or_col.index))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.small_batch_dataset,
                                                                    leftMargin=0.2, rightMargin=0.4, closureType=Closure(0),
                                                                    derivedTypeOutput=DerivedType(2), axis_param=None)
        print_and_log("Test Case 4 Passed: expected ValueError, got ValueError")

    def execute_WholeDatasetTests_checkInv_Interval_DerivedValue_ExternalDataset(self):
        """
        Execute the invariant test using the whole dataset for the function checkInv_Interval_DerivedValue
        """

        # Caso 1
        expected_df = self.rest_of_dataset.copy()
        field='liveness'

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
                                        row_or_col.iloc[i - 1] if np.issubdtype(type(value), np.number) and 0.2 < value < 0.4 and i > 0
                                        else value for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=1)

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(0), axis_param=None)

        most_frequent_value = expected_df.stack().value_counts().idxmax()
        expected_df = expected_df.apply(lambda col: col.apply(lambda x: most_frequent_value if np.issubdtype(type(x), np.number)
                                                                                            and 0.2 < x < 0.4 else x))

        pd.testing.assert_frame_equal(result, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        expected_df = self.rest_of_dataset.copy()

        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=self.rest_of_dataset, leftMargin=0.2,
                                                                rightMargin=0.4, closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)

        expected_df = expected_df.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(value) else
                                        row_or_col.iloc[i + 1] if np.issubdtype(type(value), np.number) and 0.2 < value < 0.4
                                                and i < len(row_or_col) - 1 else value for i, value in
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
