# Importing libraries
import unittest
import numpy as np
import pandas as pd
from tqdm import tqdm

# Importing functions and classes from packages
from functions.contract_invariants import ContractsInvariants
from helpers.enumerations import Belong, Operator, Closure, DataType, DerivedType, SpecialType, Operation
from helpers.logger import print_and_log


class InvariantSimpleTest(unittest.TestCase):
    """
    Class to test the contracts with simple test cases

    Attributes:
    pre_post (InvariantSimpleTest): instance of the class InvariantSimpleTest

    Methods:
    """

    #TODO: Add the remaining simple tests and adapt the existing ones to the field values

    def __init__(self):
        """
        Constructor of the class

        Attributes:
        invariants (InvariantSimpleTest): instance of the class InvariantSimpleTest
        """
        self.invariants = ContractsInvariants()

    def execute_All_SimpleTests(self):
        """
        Method to execute all simple tests of the functions of the class
        """
        simple_test_methods = [
            self.execute_CheckInv_FixValue_FixValue,
            self.execute_checkInv_FixValue_DerivedValue,
            self.execute_CheckInv_FixValue_NumOp,
            self.execute_CheckInv_Interval_FixValue,
            self.execute_CheckInv_Interval_DerivedValue,
            self.execute_CheckInv_Interval_NumOp,
            self.execute_CheckInv_SpecialValue_FixValue,
            self.execute_CheckInv_SpecialValue_DerivedValue,
            self.execute_CheckInv_SpecialValue_NumOp
        ]

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("------ STARTING INVARIANT SIMPLE TEST CASES ------")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for simple_test_method in tqdm(simple_test_methods, desc="Running Simple Tests", unit="test"):
            simple_test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("- SIMPLE INVARIANT TEST CASES EXECUTION FINISHED -")
        print_and_log("--------------------------------------------------")
        print_and_log("")

    def execute_CheckInv_FixValue_FixValue(self):
        """
        Execute the simple tests of the function CheckInv_FixValue_FixValue
        """
        print_and_log("Testing CheckInv_FixValue_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Comprobar la invariante: cambiar el valor fijo 2 por el valor fijo 999
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [5, 4, 3, 2, 1]})
        # Definir el valor fijo y la condición para el cambio
        fixValueOutput = 999
        # Aplicar la invariante
        result_df = self.invariants.checkInv_FixValue_FixValue(df, dataTypeInput=DataType(2), fixValueInput=2,
                                                               dataTypeOutput=DataType(6),
                                                               fixValueOutput=fixValueOutput)
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [0, 1, fixValueOutput, 3, 4], 'B': [5, 4, 3, fixValueOutput, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fixValueOutput = pd.to_datetime('2021-01-01')
        # Aplicar la invariante
        result_df = self.invariants.checkInv_FixValue_FixValue(df, dataTypeInput=DataType(0), fixValueInput='Clara',
                                                               dataTypeOutput=DataType(3),
                                                               fixValueOutput=fixValueOutput)
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [fixValueOutput, 'Ana', fixValueOutput, fixValueOutput, fixValueOutput],
                                    'B': [fixValueOutput, fixValueOutput, 'Ana', 'Ana', 'Ana']})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor fijo de tipo TIME 2021-01-01 por el valor fijo de tipo boolean True
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-09-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01')],
                           'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-08-01')]})
        # Definir el valor fijo y la condición para el cambio
        fixValueOutput = True
        # Aplicar la invariante
        result_df = self.invariants.checkInv_FixValue_FixValue(df, dataTypeInput=DataType(1), fixValueInput=pd.to_datetime('2021-01-01'),
                                                                dataTypeOutput=DataType(4),
                                                                fixValueOutput=fixValueOutput)
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [True, pd.to_datetime('2021-09-01'), True, True, True],
                                    'B': [True, True, True, True, pd.to_datetime('2021-08-01')]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor fijo string 'Clara' por el valor fijo de tipo FLOAT 3.0
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fixValueOutput = 3.0
        # Aplicar la invariante
        result_df = self.invariants.checkInv_FixValue_FixValue(df, dataTypeInput=DataType(0), fixValueInput='Clara',
                                                                dataTypeOutput=DataType(6),
                                                                fixValueOutput=fixValueOutput)
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar el valor fijo de tipo FLOAT 3.0 por el valor fijo de tipo STRING 'Clara'
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 2.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fixValueOutput = 'Clara'
        # Aplicar la invariante
        result_df = self.invariants.checkInv_FixValue_FixValue(df, fixValueInput=3.0, fixValueOutput=fixValueOutput)
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': ['Clara', 2.0, 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 2.0, 2.0, 2.0]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result_df, expected_df)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_FixValue_DerivedValue(self):
        """
        Execute the simple tests of the function checkInv_FixValue_DerivedValue
        """
        """
        DerivedTypes:
            0: Most Frequent
            1: Previous
            2: Next
        axis_param:
            0: Columns
            1: Rows
            None: All
        """
        print_and_log("Testing checkInv_FixValue_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor derivado 0 (Most Frequently)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), fixValueInput=0,
                                                                derivedTypeOutput=DerivedType(0), axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor derivado 2 (Previous) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), fixValueInput=5,
                                                                derivedTypeOutput=DerivedType(1), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 4], 'C': [1, 2, 3, 4, 3]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor derivado 3 (Next) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), fixValueInput=0,
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
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), fixValueInput=5,
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
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), fixValueInput=5,
                                                                derivedTypeOutput=DerivedType(0), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 8], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 7, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor previo a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), fixValueInput=5,
                                                                derivedTypeOutput=DerivedType(1), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 3, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 4, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(2), fixValueInput=5,
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 6, 6, 8, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, "Ainhoa", "Ainhoa", 5], 'B': [1, 8, "Ainhoa", 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"], 'D': [4, 5, 6, 5, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(0), fixValueInput="Ainhoa",
                                                                derivedTypeOutput=DerivedType(2), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, "Ainhoa", 5, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"], 'D': [4, 5, 6, 5, 8]})
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
        datadic = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", "Ana", pd.to_datetime('2021-01-01')], 'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')], 'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"], 'D': [pd.to_datetime('2021-01-01'), 5, "Ana", 5, 8]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_DerivedValue(dataDictionary=datadic, dataTypeInput=DataType(0), fixValueInput="Ana",
                                                                derivedTypeOutput=DerivedType(0), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01')], 'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')], 'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"], 'D': [pd.to_datetime('2021-01-01'), 5, 5, 5, 8]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckInv_FixValue_NumOp(self):
        """
        Execute the simple tests of the function checkInv_FixValue_NumOp
        """
        """
        Operation:
            0: Interpolation
            1: Mean
            2: Median
            3: Closest
        Axis:
            0: Columns
            1: Rows
            None: All
        """
        print_and_log("Testing checkInv_FixValue_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor de operación 0 (Interpolación) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
                                                          fixValueInput=0, numOpOutput=Operation(0), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [2, 3, 6, 5.5, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'int64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Case 2
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor de operación 0 (Interpolación) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
                                                            fixValueInput=0, numOpOutput=Operation(0), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [2, 2, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
                                                            fixValueInput=0, numOpOutput=Operation(1), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [(0+ 2+ 3+ 4 + 5) / 5, 2, 3, 4, 5], 'B': [2, 3, 6, (2+ 3+ 6+ 5+0) / 5, 5], 'C': [1, (1+ 0+ 3+ 4 + 5) / 5, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
                                                            fixValueInput=0, numOpOutput=Operation(1), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [(2+ 1) / 3, 2, 3, 4, 5], 'B': [2, (2+2)/3, 6, (4+4)/3, 5], 'C': [1, 2, 3, 4, (5+5)/3]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor de operación 2 (Median) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
                                                            fixValueInput=0, numOpOutput=Operation(2), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 2, 3, 4, 5], 'B': [2, 3, 6, 3, 5], 'C': [1, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor de operación 2 (Median) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
                                                            fixValueInput=0, numOpOutput=Operation(2), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 2, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor de operación 3 (Closest) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
                                                            fixValueInput=0, numOpOutput=Operation(3), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [3, 3, 6, 3, 5], 'C': [1, 1, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: cambiar el valor fijo 0 por el valor de operación 3 (Closest) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_FixValue_NumOp(dataDictionary=datadic, dataTypeInput=DataType(2),
                                                            fixValueInput=0, numOpOutput=Operation(3), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckInv_Interval_FixValue(self):
        """
        Execute the simple tests of the function checkInv_Interval_FixValue
        """
        print_and_log("Testing checkInv_Interval_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                            closureType=Closure(0), dataTypeOutput=DataType(0),
                                                            fixValueOutput='Suspenso')
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")


        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                            closureType=Closure(1), dataTypeOutput=DataType(0),
                                                            fixValueOutput='Suspenso')
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")


        # Caso 3
        # Comprobar la invariante: cambiar el rango de valores [0, 5) por el valor fijo 'Suspenso'
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                            closureType=Closure(2), dataTypeOutput=DataType(0), fixValueOutput='Suspenso')
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")


        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                            closureType=Closure(3), dataTypeOutput=DataType(0),
                                                            fixValueOutput='Suspenso')
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        field='A'
        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                            closureType=Closure(0), dataTypeOutput=DataType(0),
                                                            fixValueOutput='Suspenso', field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        field='A'
        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                            closureType=Closure(1), dataTypeOutput=DataType(0),
                                                            fixValueOutput='Suspenso', field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        field='A'
        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                            closureType=Closure(2), dataTypeOutput=DataType(0),
                                                            fixValueOutput='Suspenso', field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        field='A'
        result = self.invariants.checkInv_Interval_FixValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                            closureType=Closure(3), dataTypeOutput=DataType(0),
                                                            fixValueOutput='Suspenso', field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckInv_Interval_DerivedValue(self):
        """
        Execute the simple tests of the function checkInv_Interval_DerivedValue
        """
        print_and_log("Testing checkInv_Interval_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                closureType=Closure(0), derivedTypeOutput=DerivedType(0),
                                                                axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 2, 6, 0, 5], 'C': [0, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                closureType=Closure(3), derivedTypeOutput=DerivedType(0),
                                                                axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 0], 'B': [2, 2, 6, 2, 2], 'C': [1, 1, 1, 1, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                closureType=Closure(2), derivedTypeOutput=DerivedType(0),
                                                                axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 2, 2, 5], 'B': [2, 2, 6, 2, 5], 'C': [2, 2, 2, 2, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5, closureType=Closure(1),
                                                                derivedTypeOutput=DerivedType(1), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 4], 'B': [2, 2, 6, 0, 0], 'C': [1, 1, 2, 3, 4]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                    closureType=Closure(1),
                                                                    derivedTypeOutput=DerivedType(1), axis_param=None)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                closureType=Closure(0),
                                                                derivedTypeOutput=DerivedType(2), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 6, 0, 5], 'B': [1, 2, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                    closureType=Closure(1),
                                                                    derivedTypeOutput=DerivedType(2), axis_param=None)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='T'
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                    closureType=Closure(1), derivedTypeOutput=DerivedType(2),
                                                                    axis_param=None, field=field)
        print_and_log("Test Case 8 Passed: expected ValueError, got ValueError")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='A'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                closureType=Closure(0), derivedTypeOutput=DerivedType(0),
                                                                axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='A'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                closureType=Closure(0), derivedTypeOutput=DerivedType(1),
                                                                axis_param=0, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='A'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_DerivedValue(dataDictionary=datadic, leftMargin=0, rightMargin=5,
                                                                closureType=Closure(0), derivedTypeOutput=DerivedType(2),
                                                                axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 4, 5, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckInv_Interval_NumOp(self):
        """
        Execute the simple tests of the function checkInv_Interval_NumOp
        """
        print_and_log("Testing checkInv_Interval_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Comprobar la invariante: cambiar el rango de valores (2, 4] por el valor de operación 0 (Interpolación) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
                                                            closureType=Closure(1), numOpOutput=Operation(0), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 4, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
                                                            closureType=Closure(3), numOpOutput=Operation(0), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, np.NaN, 6, 0, 5], 'B': [0.5, np.NaN, 6, 0, 5], 'C': [1, np.NaN, 6, 0, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
                                                                closureType=Closure(3), numOpOutput=Operation(0), axis_param=None)
        print_and_log("Test Case 3 Passed: expected ValueError, got ValueError")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=0, rightMargin=3,
                                                            closureType=Closure(0), numOpOutput=Operation(1), axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 4, 5], 'B': [3, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=0, rightMargin=3,
                                                            closureType=Closure(0), numOpOutput=Operation(1), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 3, 4, 5], 'B': [3.2, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=0, rightMargin=3,
                                                            closureType=Closure(0), numOpOutput=Operation(1), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 7/3, 3, 4, 5], 'B': [1, 3, 6, 0, 5], 'C': [1, 7/3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=0, rightMargin=3,
                                                            closureType=Closure(2), numOpOutput=Operation(2), axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 3, 3, 4, 5], 'B': [3, 3, 6, 3, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=0, rightMargin=3,
                                                            closureType=Closure(2), numOpOutput=Operation(2), axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=0, rightMargin=4,
                                                            closureType=Closure(0), numOpOutput=Operation(3), axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 1, 2, 4, 5], 'B': [1, 2, 6, 0, 5], 'C': [0, 1, 2, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 9 Passed: the function returned the expected dataframe")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=0, rightMargin=4,
                                                            closureType=Closure(0), numOpOutput=Operation(3), axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 4, 5], 'B': [3, 2, 6, 0, 5], 'C': [2, 1, 2, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='T'
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
                                                                closureType=Closure(3), numOpOutput=Operation(0),
                                                                axis_param=None, field=field)
        print_and_log("Test Case 11 Passed: expected ValueError, got ValueError")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='A'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
                                                            closureType=Closure(3), numOpOutput=Operation(0),
                                                            axis_param=None, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 1.25, 2.5, 3.75, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 12 Passed: the function returned the expected dataframe")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='A'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
                                                            closureType=Closure(3), numOpOutput=Operation(1),
                                                            axis_param=None, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 2.8, 2.8, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='A'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
                                                            closureType=Closure(3), numOpOutput=Operation(2),
                                                            axis_param=None, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 14 Passed: the function returned the expected dataframe")

        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field='A'
        # Aplicar la invariante
        result = self.invariants.checkInv_Interval_NumOp(dataDictionary=datadic, leftMargin=2, rightMargin=4,
                                                            closureType=Closure(3), numOpOutput=Operation(3),
                                                            axis_param=None, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckInv_SpecialValue_FixValue(self):
        """
        Execute the simple tests of the function checkInv_SpecialValue_FixValue
        """
        print_and_log("Testing checkInv_SpecialValue_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                dataTypeOutput=DataType(2), fixValueOutput=999,
                                                                missing_values=missing_values, axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 999], 'B': [2, 999, 4, 999, 10], 'C': [999, 10, 999, 4, 999], 'D': [2, 999, 4, 6, 10],
             'E': [999, 10, 999, 4, 999]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Comprobar la invariante: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                dataTypeOutput=DataType(2), fixValueOutput=999,
                                                                missing_values=missing_values, axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 999], 'B': [2, 999, 4, np.NaN, 10], 'C': [999, 10, 999, 4, 999], 'D': [2, None, 4, 6, 10],
             'E': [999, 10, 999, 4, 999]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Comprobar la invariante: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                dataTypeOutput=DataType(2), fixValueOutput=999,
                                                                axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 999], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 999, 3, 4, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                dataTypeOutput=DataType(2), fixValueOutput=999,
                                                                axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 999, 3, 4, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")


        # Caso 5
        # Comprobar la invariante: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10], 'E': [1, 10, 3, 4, 1]})
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                dataTypeOutput=DataType(2), fixValueOutput=999, axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 999], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 999], 'E': [1, 10, 3, 4, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 5 Passed: the function returned the expected dataframe")

        # Caso 6
        # Comprobar la invariante: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field= 'B'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                dataTypeOutput=DataType(2), fixValueOutput=999,
                                                                missing_values=missing_values, axis_param=0, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, 999, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Comprobar la invariante: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field= 'B'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                dataTypeOutput=DataType(2), fixValueOutput=999,
                                                                missing_values=missing_values, axis_param=0, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Comprobar la invariante: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        field= 'C'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                dataTypeOutput=DataType(2), fixValueOutput=999,
                                                                axis_param=None, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # ValueError
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        field= 'T'
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_FixValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                    dataTypeOutput=DataType(2), fixValueOutput=999,
                                                                    axis_param=None, field=field)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_CheckInv_SpecialValue_DerivedValue(self):
        """
        Execute the simple tests of the function checkInv_SpecialValue_DerivedValue
        """
        print_and_log("Testing checkInv_SpecialValue_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 1 Passed: the function returned the expected dataframe")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 2 Passed: the function returned the expected dataframe")

        # Caso 3
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 3, 3], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [3, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 3 Passed: the function returned the expected dataframe")

        # Caso 4
        # Comprobar la invariante: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 4 Passed: the function returned the expected dataframe")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 3, 3], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [3, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 6 Passed: the function returned the expected dataframe")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                    derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                    axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, np.NaN, 3, 4], 'B': [2, 2, 3, 12, 12], 'C': [10, 0, 0, 3, 2], 'D': [1, 8, 8, 8, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 7 Passed: the function returned the expected dataframe")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                    derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                    axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, np.NaN, 3, 12, 12], 'C': [10, 0, 4, 12, 2], 'D': [10, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 8 Passed: the function returned the expected dataframe")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                        derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                        axis_param=None)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                    derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                    axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 4, 1, 1], 'B': [2, 4, 12, 12, 12], 'C': [10, 0, 3, 2, 2], 'D': [8, 8, 8, 2, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 10 Passed: the function returned the expected dataframe")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                    derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                    axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 4, 12, 12], 'B': [2, 0, 3, 12, 12], 'C': [10, 0, 8, 1, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 11 Passed: the function returned the expected dataframe")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                        derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                        axis_param=None)
        print_and_log("Test Case 12 Passed: expected ValueError, got ValueError")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic,specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(0),missing_values=missing_values,
                                                                    axis_param=None)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 3, 3], 'C': [3, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 13 Passed: the function returned the expected dataframe")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic,specialTypeInput=SpecialType(2),
                                                                        derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                        axis_param=None)
        print_and_log("Test Case 14 Passed: expected ValueError, got ValueError")

        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic,specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(0),missing_values=missing_values,
                                                                    axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [3, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 15 Passed: the function returned the expected dataframe")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic,specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(0),missing_values=missing_values,
                                                                    axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 4, 2], 'C': [0, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 16 Passed: the function returned the expected dataframe")

        # Caso 17
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                    axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 17 Passed: the function returned the expected dataframe")

        # Caso 18
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                    axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 4, 1], 'C': [2, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 18 Passed: the function returned the expected dataframe")

        # Caso 19
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                    axis_param=0)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 19 Passed: the function returned the expected dataframe")

        # Caso 20
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                    axis_param=1)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 3, 2], 'C': [1, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 20 Passed: the function returned the expected dataframe")


        # Caso 21
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field= 'T'
        # Aplicar la invariante
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                        derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                        axis_param=None, field=field)
        print_and_log("Test Case 21 Passed: expected ValueError, got ValueError")

        # Caso 22
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field= 'A'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 0], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 22 Passed: the function returned the expected dataframe")

        # Caso 23
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field= 'A'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 0, 0, 0], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 23 Passed: the function returned the expected dataframe")

        # Caso 24
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = None
        field= 'A'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                    derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 24 Passed: the function returned the expected dataframe")

        # Caso 25
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field= 'A'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                    derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, np.NaN, 3, 4], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 25 Passed: the function returned the expected dataframe")

        # Caso 26
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = None
        field= 'A'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                    derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 26 Passed: the function returned the expected dataframe")

        # Caso 27
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field= 'A'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                    derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 4, 1, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 27 Passed: the function returned the expected dataframe")

        # Caso 28
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field='C'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(0), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [3, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 28 Passed: the function returned the expected dataframe")

        # Caso 29
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field='C'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(1), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 29 Passed: the function returned the expected dataframe")

        # Caso 30
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field='C'
        # Aplicar la invariante
        result = self.invariants.checkInv_SpecialValue_DerivedValue(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                    derivedTypeOutput=DerivedType(2), missing_values=missing_values,
                                                                    axis_param=1, field=field)
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })

        # Verificar si el resultado obtenido coincide con el esperado
        pd.testing.assert_frame_equal(result, expected)
        print_and_log("Test Case 30 Passed: the function returned the expected dataframe")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")



    def execute_CheckInv_SpecialValue_NumOp(self):
        """
        Execute the simple tests of the function checkInv_SpecialValue_NumOp
        """
        """
        SpecialTypes:
            0: Missing
            1: Invalid
            2: Outlier
        Operation:
            0: Interpolation
            1: Mean
            2: Median
            3: Closest
        Axis:
            0: Columns
            1: Rows
            None: All
        """
        print_and_log("Testing checkInv_SpecialValue_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

        #MISSING
        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 2.0, 2], 'B': [2, 2 + 4/3, 2 + 8/3, 6, 12], 'C': [10, 7.5, 5, 2.5, 0], 'D': [8.2, 8.2, 6, 4, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                             numOpOutput=Operation(0), missing_values=missing_values,
                                                             axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 1 Passed: got the dataframe expected")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3.61, 3.61, 1], 'B': [2, 3.61, 3.61, 6, 12], 'C': [10, 1, 3.61, 3.61, 0], 'D': [1, 8.2, 6, 1, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                numOpOutput=Operation(1), missing_values=missing_values,
                                                                axis_param=None)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 3
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 3.5, 1], 'B': [2, 3, 3.5, 6, 1], 'C': [10, 2.5, 3, 3, 0], 'D': [1.5, 8.2, 6, 3.5, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                numOpOutput=Operation(2), missing_values=missing_values,
                                                                axis_param=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 2 Passed: got the dataframe expected")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 3, 3, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        missing_values = [1, 3, 4]
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                             numOpOutput=Operation(3), missing_values=missing_values,
                                                             axis_param=0)
        print_and_log("Test Case 4 Passed: Expected ValueError, got ValueError")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(0),
                                                                numOpOutput=Operation(3), missing_values=missing_values,
                                                                axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 5 Passed: got the dataframe expected")

        #Invalid
        # Caso 6
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 2.0, 2], 'B': [2, 2 + 4/3, 2 + 8/3, 6, 12], 'C': [10, 7.5, 5, 2.5, 0], 'D': [8.2, 8.2, 6, 4, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                             numOpOutput=Operation(0), missing_values=missing_values,
                                                             axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 6 Passed: got the dataframe expected")

        # Caso 7
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3.61, 3.61, 1], 'B': [2, 3.61, 3.61, 6, 12], 'C': [10, 1, 3.61, 3.61, 0], 'D': [1, 8.2, 6, 1, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                numOpOutput=Operation(1), missing_values=missing_values,
                                                                axis_param=None)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 7 Passed: got the dataframe expected")

        # Caso 8
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 3.5, 1.5], 'B': [2, 3, 3.5, 6, 12], 'C': [10, 2.5, 3, 3, 0], 'D': [1.5, 8.2, 6, 3.5, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                numOpOutput=Operation(2), missing_values=missing_values,
                                                                axis_param=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 8 Passed: got the dataframe expected")

        # Caso 9
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 0], 'B': [2, 2, 3, 6, 12], 'C': [10, 3, 4, np.NaN, 0], 'D': [2, 8.2, 2, 2, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                             numOpOutput=Operation(3), missing_values=missing_values,
                                                             axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 9 Passed: got the dataframe expected")

        # Caso 10
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})

        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(1),
                                                                numOpOutput=Operation(3), missing_values=missing_values,
                                                                axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 10 Passed: got the dataframe expected")

        # Outliers
        # Caso 11
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [1, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64',  # Convertir D a float64
            'B': 'float64', # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                numOpOutput=Operation(0), axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 11 Passed: got the dataframe expected")

        # Caso 12
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        datadic = datadic.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 3.61], 'C': [3.61, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})


        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                numOpOutput=Operation(1), missing_values=None,
                                                                axis_param=None)


        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 12 Passed: got the dataframe expected")

        # Caso 13
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        datadic = datadic.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir B a float64
        })
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 1.5], 'C': [1.5, 1, 3, 3, 0], 'D': [1, 2.5, 6, 1, 2]})
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                numOpOutput=Operation(2), missing_values=None,
                                                                axis_param=1)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 13 Passed: got the dataframe expected")

        # Caso 14
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 3, 3, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })

        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                numOpOutput=Operation(3), missing_values=None,
                                                                axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 14 Passed: got the dataframe expected")

        # Caso 15
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [10, 6, 3, 3, 0], 'D': [1, 2, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                numOpOutput=Operation(3), axis_param=0)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 15 Passed: got the dataframe expected")

        # Caso 16
        # Probamos a aplicar la operación mean sobre un field concreto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 2.84, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field = 'D'
        missing_values=[8.2]
        result_df = self.invariants.checkInv_SpecialValue_NumOp(dataDictionary=datadic, specialTypeInput=SpecialType(2),
                                                                numOpOutput=Operation(1), missing_values=missing_values,
                                                                axis_param=0, field=field)
        pd.testing.assert_frame_equal(expected_df, result_df)
        print_and_log("Test Case 16 Passed: got the dataframe expected")











