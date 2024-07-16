# Importing libraries
import math
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importing functions and classes from packages
import functions.contract_invariants as invariants
from helpers.enumerations import Closure, DataType, DerivedType, SpecialType, Operation, Belong
from helpers.logger import print_and_log


class InvariantsSimpleTest(unittest.TestCase):
    """
    Class to test the invariants with simple test cases

    Attributes:
    unittest.TestCase: class that inherits from unittest.TestCase

    Methods:
    execute_All_SimpleTests: method to execute all simple tests of the functions of the class
    execute_checkInv_FixValue_FixValue: method to execute the simple tests of the function checkInv_FixValue_FixValue
    execute_checkInv_FixValue_DerivedValue: method to execute the simple
    tests of the function checkInv_FixValue_DerivedValue
    execute_checkInv_FixValue_NumOp: method to execute the simple tests of the function checkInv_FixValue_NumOp
    execute_checkInv_Interval_FixValue: method to execute the simple tests of the function checkInv_Interval_FixValue
    execute_checkInv_Interval_DerivedValue: method to execute the simple tests
    of the function checkInv_Interval_DerivedValue
    execute_checkInv_Interval_NumOp: method to execute the simple tests of the function checkInv_Interval_NumOp
    execute_checkInv_SpecialValue_FixValue: method to execute the simple
    tests of the function checkInv_SpecialValue_FixValue
    execute_checkInv_SpecialValue_DerivedValue: method to execute the simple
    tests of the function checkInv_SpecialValue_DerivedValue
    execute_checkInv_SpecialValue_NumOp: method to execute the simple tests of the function checkInv_SpecialValue_NumOp
    """

    def __init__(self):
        """
        Constructor of the class
        """
        super().__init__()
        self.invariants = invariants

    def execute_All_SimpleTests(self):
        """
        Method to execute all simple tests of the functions of the class
        """
        simple_test_methods = [
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
        print_and_log("------ STARTING INVARIANT SIMPLE TEST CASES ------")
        print_and_log("--------------------------------------------------")
        print_and_log("")

        for simple_test_method in tqdm(simple_test_methods, desc="Running Invariant Simple Tests", unit="test"):
            simple_test_method()

        print_and_log("")
        print_and_log("--------------------------------------------------")
        print_and_log("- INVARIANT SIMPLE TEST CASES EXECUTION FINISHED -")
        print_and_log("--------------------------------------------------")
        print_and_log("")

    def execute_checkInv_FixValue_FixValue(self):
        """
        Execute the simple tests of the function checkInv_FixValue_FixValue
        """
        print_and_log("Testing checkInv_FixValue_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Ejecutar la transformación de datos: cambiar el valor fijo 2 por el valor fijo 999
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [5, 4, 3, 2, 1]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [999]
        expected_df = pd.DataFrame({'A': [0, 1, 999, 3, 4], 'B': [5, 4, 3, 999, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=[2],
                                                               data_type_output_list=None,
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(0), field_in=None, field_out=None)

        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01')],
             'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), 'Ana', 'Ana', 'Ana']})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None,
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(0), field_in=None, field_out=None)

        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo TIME 2021-01-01 por el valor fijo de tipo boolean True
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-09-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01')],
                           'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-08-01')]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [True]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [True, pd.to_datetime('2021-09-01'), True, True, True],
                                    'B': [True, True, True, True, pd.to_datetime('2021-08-01')]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=[pd.to_datetime('2021-01-01')],
                                                               data_type_output_list=None,
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo string 'Clara' por el valor fijo de tipo FLOAT 3.0
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None,
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo FLOAT 3.0 por el valor fijo de tipo STRING 'Clara'
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 2.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = ['Clara']
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Clara', 2.0, 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 2.0, 2.0, 2.0]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               input_values_list=[3.0],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(0), field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Test with NOTBELONG

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo 2 por el valor fijo 999
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [5, 4, 3, 2, 1]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [999]
        expected_df = pd.DataFrame({'A': [0, 1, 999, 3, 4], 'B': [5, 4, 3, 999, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=[2],
                                                               data_type_output_list=None,
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None)

        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01')],
             'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), 'Ana', 'Ana', 'Ana']})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None,
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1),
                                                               field_in=None, field_out=None)

        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo TIME 2021-01-01 por el valor fijo de tipo boolean True
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-09-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01')],
                           'B': [pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'),
                                 pd.to_datetime('2021-08-01')]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [True]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [True, pd.to_datetime('2021-09-01'), True, True, True],
                                    'B': [True, True, True, True, pd.to_datetime('2021-08-01')]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=[pd.to_datetime('2021-01-01')],
                                                               data_type_output_list=None,
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1), field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar el valor fijo string 'Clara' por el valor fijo de tipo FLOAT 3.0
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None,
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1), field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo de tipo FLOAT 3.0 por el valor fijo de tipo STRING 'Clara'
        # Crear un DataFrame de prueba
        df = pd.DataFrame({'A': [3.0, 2.0, 3.0, 3.0, 3.0], 'B': [3.0, 3.0, 2.0, 2.0, 2.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = ['Clara']
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': ['Clara', 2.0, 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 2.0, 2.0, 2.0]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               input_values_list=[3.0],
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1), field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected_df = pd.DataFrame(
            {'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-01-01')],
             'B': ['Clara', 'Clara', 'Ana', pd.to_datetime('2021-01-01'), 'Ana']})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None,
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(0), field_in=field_in, field_out=field_out)

        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar el valor fijo 'Clara' por el valor fijo de fecha 2021-01-01
        # Crear un DataFrame de prueba
        df = pd.DataFrame(
            {'A': ['Clara', 'Ana', 'Clara', 'Clara', 'Clara'], 'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [pd.to_datetime('2021-01-01')]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [pd.to_datetime('2021-01-01'), 'Ana', pd.to_datetime('2021-01-01'), 'Clara', 'Clara'],
                                    'B': ['Clara', 'Clara', 'Ana', 'Ana', 'Ana']})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara'],
                                                               data_type_output_list=None,
                                                               output_values_list=fix_value_output, belong_op_in=Belong(0),
                                                               belong_op_out=Belong(1), field_in=field_in, field_out=field_out)

        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 14]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', 14, None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 14]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 'Clara', np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Caso 15
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 14]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', 14, None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None, input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Caso 16
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 14]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 'Clara', np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, belong_op_out=Belong(1),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # Caso 17
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 'Clara', np.NaN], 'B': [3.0, 3.0, 'Ana', '8', None]})
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                                   data_type_input_list=None,
                                                                   input_values_list=['Clara', '8'],
                                                                   data_type_output_list=None, belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   output_values_list=fix_value_output)
        print_and_log("Test Case 17 Passed: Expected ValueError, got ValueError")

        # Caso 18
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', '8', None]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 3.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', 3.0, None]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['Clara', '8'],
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19
        df = pd.DataFrame({'A': ['Clara', 'Ana', 'Clara', 'Clara', np.NaN], 'B': ['Clara', 'Clara', 'Ana', 3.0, 3.0]})
        # Definir el valor fijo y la condición para el cambio
        fix_value_output = [3.0, 6.0]
        # Definir el resultado esperado
        expected_df = pd.DataFrame({'A': [3.0, 'Ana', 3.0, 3.0, np.NaN], 'B': [3.0, 3.0, 'Ana', 6.0, 6.0]})

        result = self.invariants.check_inv_fix_value_fix_value(data_dictionary_in=df, data_dictionary_out=expected_df,
                                                               data_type_input_list=None,
                                                               input_values_list=['Clara', 3.0],
                                                               data_type_output_list=None, belong_op_out=Belong(0),
                                                               field_in=None, field_out=None,
                                                               output_values_list=fix_value_output)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")


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
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 0 (Most Frequently)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=0,
                                                                   derived_type_output=DerivedType(0), axis_param=None,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor derivado 2 (Previous) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 4], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(1), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 3 (Next) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=0,
                                                                   derived_type_output=DerivedType(2), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor más frecuente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 3, 3], 'B': [1, 8, 4, 4, 4], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(0), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor más frecuente a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 7, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 8], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 7, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(0), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor previo a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 5], 'B': [1, 8, 3, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 2, 6, 4, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(1), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 6, 6, 8, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(2), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, "Ainhoa", "Ainhoa", 5], 'B': [1, 8, "Ainhoa", 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"],
             'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, "Ainhoa", 5, 5], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, "Ainhoa"], 'D': [4, 5, 6, 5, 8]})
        expected = expected.astype({
            'A': 'object',  # Convertir A a object
            'B': 'object',  # Convertir B a int64
            'C': 'object',  # Convertir C a object
            'D': 'int64'  # Convertir D a object
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=None,
                                                                   fix_value_input="Ainhoa",
                                                                   derived_type_output=DerivedType(2), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar el valor fijo "Ana" por el valor más frecuente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", "Ana", pd.to_datetime('2021-01-01')],
                                'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')],
                                'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"],
                                'D': [pd.to_datetime('2021-01-01'), 5, "Ana", 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, pd.to_datetime('2021-01-01'), "Ainhoa", pd.to_datetime('2021-01-01'),
                                       pd.to_datetime('2021-01-01')],
                                 'B': [pd.to_datetime('2021-01-01'), 8, "Ainhoa", 4, pd.to_datetime('2021-01-01')],
                                 'C': [1, pd.to_datetime('2021-01-01'), 3, 4, "Ainhoa"],
                                 'D': [pd.to_datetime('2021-01-01'), 5, 5, 5, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=None,
                                                                   fix_value_input="Ana",
                                                                   derived_type_output=DerivedType(0), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 0 (Most Frequently)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 5, 5], 'B': [1, 2, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=0,
                                                                   derived_type_output=DerivedType(0), axis_param=None,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor derivado 2 (Previous) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 5, 5], 'B': [1, 8, 4, 4, 5], 'C': [1, 2, 3, 4, 3]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 4], 'B': [1, 8, 4, 4, 3], 'C': [1, 2, 3, 4, 3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(1), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor derivado 3 (Next) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=0,
                                                                   derived_type_output=DerivedType(2), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")


        # Caso 15
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor previo a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 3, 3], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 4, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(1), axis_param=1,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Ejecutar la transformación de datos: cambiar el valor fijo 5 por el valor siguiente a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 5], 'B': [1, 8, 5, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 5, 6, 5, 8]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ["0", 2, 3, 3, 4], 'B': [1, 8, 3, 4, 3], 'C': [1, 2, 3, 4, 8], 'D': [4, 3, 6, 3, 8]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                   data_type_input=DataType(2),
                                                                   fix_value_input=5,
                                                                   derived_type_output=DerivedType(2), axis_param=0,
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   field_in=None, field_out=None,
                                                                   data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_FixValue_NumOp(self):
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
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 0 (Interpolación) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [1, 0, 0, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 5.5, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'int64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(0), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [(0 + 2 + 3 + 4 + 5) / 5, 2, 3, 4, 5], 'B': [2, 3, 6, (2 + 3 + 6 + 5 + 0) / 5, 5],
                                 'C': [1, (1 + 0 + 3 + 4 + 5) / 5, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(1), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [(2 + 1) / 3, 2, 3, 4, 5], 'B': [2, (2 + 2) / 3, 6, (4 + 4) / 3, 5], 'C': [1, 2, 3, 4, (5 + 5) / 3]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(1), axis_param=1,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 2 (Median) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 2, 3, 4, 5], 'B': [2, 3, 6, 3, 5], 'C': [1, 3, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(2), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 2 (Median) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 2, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(2), axis_param=1,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 3 (Closest) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 3, 4, 5], 'B': [3, 3, 6, 3, 5], 'C': [1, 1, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(3), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 3 (Closest) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(3), axis_param=1,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 0 (Interpolación) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [1, 0, 0, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 5.5, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'int64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(0), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 0, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(1), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 1 (Mean) a nivel de fila
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 0, 6, 0, 5], 'C': [1, 2, 3, 4, 0]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [4, 2, 3, 4, 5], 'B': [2, 4, 6, 7, 5], 'C': [1, 2, 3, 4, 9]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(1), axis_param=1,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor fijo 0 por el valor de operación 2 (Median) a nivel de columna
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 0, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 2, 3, 4, 5], 'B': [2, 3, 6, 3, 5], 'C': [1, 3, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_fix_value_num_op(data_dictionary_in=datadic.copy(), data_type_input=DataType(2),
                                                            fix_value_input=0, num_op_output=Operation(2), axis_param=0,
                                                            belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                            field_in=None, field_out=None,
                                                            data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_Interval_FixValue(self):
        """
        Execute the simple tests of the function checkInv_Interval_FixValue
        """
        print_and_log("Testing checkInv_Interval_FixValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # _________________________________________BELONG-BELONG_________________________________________________________
        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(0), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(1), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el rango de valores [0, 5) por el valor fijo 'Suspenso'
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(2), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(3), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 5, 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(0), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0),
                                                              belong_op_out=Belong(0), data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'so', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(1), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'penso', 'Suspenso', 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(2), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
             'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(3), data_type_output=DataType(0),
                                                              fix_value_output='Suspe', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 8 Failed: Expected False, but got True"
        print_and_log("Test Case 8 Passed: Expected False, got False")

        # _________________________________________BELONG-NOTBELONG_________________________________________________________
        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(0), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 0, 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(1), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 10 Failed: Expected False, but got True"
        print_and_log("Test Case 10 Passed: Expected False, got False")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el rango de valores [0, 5) por el valor fijo 'Suspenso'
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 5],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(2), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso'],
                                 'B': ['Suspenso', 'Suspenso', 6, 'Suspenso', 'Suspenso'],
                                 'C': ['Suspenso', 'Suspenso', 'Suspenso', 'Suspenso', 'Suspenso']})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(3), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', belong_op_in=Belong(0),
                                                              field_in=None, field_out=None,
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'Spenso', 'Suspenso', 'Suspenso', 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(0), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0),
                                                              belong_op_out=Belong(1), data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 'uspenso', 'Suspenso', 'Suspenso', 'Suspenso'], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(1), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': ['Repite', 'Aprobado', 5, 14, 5], 'B': [2, 3, 6, 0, 5],
                                 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(2), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': ['', 0, '', '', 4], 'B': [2, 3, 6, 0, 5],
             'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        field_in = 'A'
        field_out = 'A'
        result = self.invariants.check_inv_interval_fix_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                              right_margin=5,
                                                              closure_type=Closure(3), data_type_output=DataType(0),
                                                              fix_value_output='Suspenso', field_in=field_in,
                                                              field_out=field_out,
                                                              belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                              data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_Interval_DerivedValue(self):
        """
        Execute the simple tests of the function checkInv_Interval_DerivedValue
        """
        print_and_log("Testing checkInv_Interval_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 2, 6, 0, 5], 'C': [0, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=1,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 0], 'B': [2, 2, 6, 2, 2], 'C': [1, 1, 1, 1, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(3),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=0,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [2, 2, 2, 2, 5], 'B': [2, 2, 6, 2, 5], 'C': [2, 2, 2, 2, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(2),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=None,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 4], 'B': [2, 2, 6, 0, 0], 'C': [1, 1, 2, 3, 4]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(1),
                                                                  derived_type_output=DerivedType(1), axis_param=0,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  field_in=None, field_out=None,
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                      right_margin=5,
                                                                      closure_type=Closure(1),
                                                                      derived_type_output=DerivedType(1), axis_param=None,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      field_in=None, field_out=None,
                                                                      data_dictionary_out=expected)
        print_and_log("Test Case 5 Passed: expected ValueError, got ValueError")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                      right_margin=5,
                                                                      closure_type=Closure(1),
                                                                      derived_type_output=DerivedType(2), axis_param=None,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      field_in=None, field_out=None,
                                                                      data_dictionary_out=expected)
        print_and_log("Test Case 6 Passed: expected ValueError, got ValueError")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                      right_margin=5,
                                                                      closure_type=Closure(1),
                                                                      derived_type_output=DerivedType(2),
                                                                      axis_param=None, field_in=field_in, field_out=field_out,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                      data_dictionary_out=expected)
        print_and_log("Test Case 7 Passed: expected ValueError, got ValueError")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 0, 0, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=1, field_in=field_in, field_out=field_out,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(1),
                                                                  axis_param=0, field_in=field_in, field_out=field_out,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 4, 5, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(2),
                                                                  axis_param=1, field_in=field_in, field_out=field_out,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 2, 6, 0, 5], 'C': [0, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=1,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(3),
                                                                  derived_type_output=DerivedType(0),
                                                                  axis_param=0,
                                                                  field_in=None, field_out=None,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [0, 3, 6, 0, 5], 'C': [0, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(1),
                                                                  derived_type_output=DerivedType(1), axis_param=0,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                  field_in=None, field_out=None,
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                      right_margin=5,
                                                                      closure_type=Closure(1),
                                                                      derived_type_output=DerivedType(2), axis_param=None,
                                                                      belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                      field_in=None, field_out=None,
                                                                      data_dictionary_out=expected)
        print_and_log("Test Case 16 Passed: expected ValueError, got ValueError")


        # Caso 19
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 0, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_derived_value(data_dictionary_in=datadic.copy(), left_margin=0,
                                                                  right_margin=5,
                                                                  closure_type=Closure(0),
                                                                  derived_type_output=DerivedType(1),
                                                                  axis_param=0, field_in=field_in, field_out=field_out,
                                                                  belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                  data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_Interval_NumOp(self):
        """
        Execute the simple tests of the function checkInv_Interval_NumOp
        """
        print_and_log("Testing checkInv_Interval_NumOp Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # ------------------------------------------BELONG-BELONG----------------------------------------------
        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 8], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 4, 6, 8], 'B': [2, 4, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(1), num_op_output=Operation(0),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 6, 0, 5], 'B': [2, 2, 6, 0, 5], 'C': [1, 2, 6, 0, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=3,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 4, 5], 'B': [3, 3, 6, 0, 5], 'C': [1, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(0),
                                                           field_in=None, field_out=None,
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 3 Failed: Expected False, but got True"
        print_and_log("Test Case 3 Passed: Expected False, got False")

        # Caso 4
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 3, 4, 5], 'B': [3.2, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 7 / 4, 3, 4, 5], 'B': [1, 3, 6, 0, 5], 'C': [1, 7 / 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 5 Failed: Expected False, but got True"
        print_and_log("Test Case 5 Passed: Expected False, got False")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 3, 3, 4, 5], 'B': [3, 3, 6, 3, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           field_in=None, field_out=None,
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 1, 2, 4, 5], 'B': [1, 2, 6, 0, 5], 'C': [0, 1, 2, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           field_in=None, field_out=None,
                                                           belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 4, 5], 'B': [3, 2, 6, 0, 5], 'C': [2, 3, 2, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=field_in, field_out=field_out,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 10 Passed: expected ValueError, got ValueError")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 16], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 4, 8, 11, 16], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 11 Failed: Expected False, but got True"
        print_and_log("Test Case 11 Passed: Expected False, got False")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 2.8, 2.8, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # ------------------------------------------BELONG-NOTBELONG----------------------------------------------
        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 8], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 4, 6, 8], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(1), num_op_output=Operation(0),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 6, 0, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 6, 0, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la invariante
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=3,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # Caso 17
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 3, 4, 5], 'B': [3, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           belong_op_out=Belong(1),
                                                           field_in=None, field_out=None,
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Caso 18
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 3, 4, 5], 'B': [3.1, 3, 6, 0, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=0,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 7 / 3, 3, 4, 5], 'B': [1, 3, 6, 0, 5], 'C': [1, 7 / 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(0), num_op_output=Operation(1),
                                                           axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Caso 20
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [3, 3, 3, 4, 5], 'B': [2, 3, 6, 3, 5], 'C': [3, 3, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3,
                                                           closure_type=Closure(2), num_op_output=Operation(2),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           field_in=None, field_out=None,
                                                           belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 3, 6, 4, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=3, closure_type=Closure(2),
                                                           num_op_output=Operation(2), axis_param=1,
                                                           field_in=None, field_out=None,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Caso 22
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 1, 2, 4, 5], 'B': [3, 2, 6, 0, 5], 'C': [0, 1, 2, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=4,
                                                           closure_type=Closure(0), num_op_output=Operation(3),
                                                           axis_param=None, belong_op_in=Belong(0),
                                                           field_in=None, field_out=None,
                                                           belong_op_out=Belong(1), data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 3, 2, 4, 5], 'B': [3, 2, 6, 0, 5], 'C': [2, 1, 2, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=0,
                                                           right_margin=4, closure_type=Closure(0),
                                                           num_op_output=Operation(3), axis_param=0,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 23 Failed: Expected False, but got True"
        print_and_log("Test Case 23 Passed: Expected False, got False")

        # Caso 24
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                               right_margin=4,
                                                               closure_type=Closure(3), num_op_output=Operation(0),
                                                               axis_param=None, field_in=field_in, field_out=field_out,
                                                               belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                               data_dictionary_out=expected)
        print_and_log("Test Case 24 Passed: expected ValueError, got ValueError")

        # Caso 25
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 16], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 4, 8, 12, 16], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})

        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(0),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.9, 2.8, 2.8, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(1),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

        # Caso 27
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2.8, 3, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(2),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 27 Failed: Expected True, but got False"
        print_and_log("Test Case 27 Passed: Expected True, got True")

        # Caso 28
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame({'A': [0, 2, 3, 4, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame({'A': [0, 2, 2, 3, 5], 'B': [2, 3, 6, 0, 5], 'C': [1, 2, 3, 4, 5]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_interval_num_op(data_dictionary_in=datadic.copy(), left_margin=2,
                                                           right_margin=4,
                                                           closure_type=Closure(3), num_op_output=Operation(3),
                                                           axis_param=None, field_in=field_in, field_out=field_out,
                                                           belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                           data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_SpecialValue_FixValue(self):
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
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 999], 'B': [2, 999, 4, 999, 10], 'C': [999, 10, 999, 4, 999], 'D': [2, 999, 4, 6, 10],
             'E': [999, 10, 999, 4, 999]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(0),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0),
                                                                   fix_value_output=999, missing_values=missing_values,
                                                                   axis_param=0, field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 999], 'B': [2, 999, 4, np.NaN, 10], 'C': [999, 10, 999, 4, 999],
             'D': [2, None, 4, 6, 10],
             'E': [999, 10, 999, 4, 999]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(1),
                                                                   data_type_output=DataType(2),
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 999], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 999, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2),
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                   fix_value_output=999,
                                                                   axis_param=None, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 999, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   axis_param=0, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, -1], 'B': [2, 3, 4, 6, 0], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, -1], 'B': [2, 3, 4, 6, 0], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   axis_param=1, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, 999, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(0),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(1),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(0), fix_value_output=999,
                                                                   axis_param=None, field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 999], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(0),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1),
                                                                   fix_value_output=999, missing_values=missing_values,
                                                                   axis_param=0, field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 999, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(1),
                                                                   data_type_output=DataType(2),
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   fix_value_output=999, missing_values=missing_values,
                                                                   axis_param=0, field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 999, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2),
                                                                   belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                   fix_value_output=999, axis_param=None,
                                                                   field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   axis_param=0, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 12 Failed: Expected True, but got False"
        print_and_log("Test Case 12 Passed: Expected True, got True")

        # Caso 13
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 0], 'C': [1, 10, 3, 4, -1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, -3]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 0], 'C': [1, 10, 3, 4, -1], 'D': [2, 3, 4, 6, 999],
             'E': [1, 10, 3, 4, -3]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   axis_param=1, field_in=None, field_out=None)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 13 Failed: Expected False, but got True"
        print_and_log("Test Case 13 Passed: Expected False, got False")

        # Caso 14
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(0),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # Caso 15
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        missing_values = [1, 3]
        field_in = 'B'
        field_out = 'B'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 999, 4, np.NaN, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, None, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})

        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(1),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   missing_values=missing_values, axis_param=0,
                                                                   field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 15 Failed: Expected False, but got True"
        print_and_log("Test Case 15 Passed: Expected False, got False")

        # Caso 16
        # Ejecutar la transformación de datos: cambiar el valor especial 2 (Outliers) a nivel de fila por el valor fijo 999
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 10, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [1, 999, 3, 4, 1], 'D': [2, 3, 4, 6, 10],
             'E': [1, 10, 3, 4, 1]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_fix_value(data_dictionary_in=datadic.copy(),
                                                                   data_dictionary_out=expected,
                                                                   special_type_input=SpecialType(2),
                                                                   data_type_output=DataType(2), belong_op_in=Belong(0),
                                                                   belong_op_out=Belong(1), fix_value_output=999,
                                                                   axis_param=None, field_in=field_in, field_out=field_out)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_SpecialValue_DerivedValue(self):
        """
        Execute the simple tests of the function checkInv_SpecialValue_DerivedValue
        """
        print_and_log("Testing checkInv_SpecialValue_DerivedValue Function")
        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Caso 1
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 3, 3], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [3, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=None, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Ejecutar la transformación de datos: cambiar el valor especial 1 (Invalid) a nivel de columna por el valor derivado 0 (Most Frequent)
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 3, 3], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [3, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=None, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, np.NaN, 3, 4], 'B': [2, 2, 3, 12, 12], 'C': [10, 0, 0, 3, 2], 'D': [1, 8, 8, 8, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(1),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 7 Failed: Expected True, but got False"
        print_and_log("Test Case 7 Passed: Expected True, got True")

        # Caso 8
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, np.NaN, 3, 12, 12], 'C': [10, 0, 4, 12, 2], 'D': [10, 8, 8, 3, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(1),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                           special_type_input=SpecialType(1),
                                                                           derived_type_output=DerivedType(1),
                                                                           missing_values=missing_values,
                                                                           axis_param=None, belong_op_in=Belong(0),
                                                                           field_in=None, field_out=None,
                                                                           belong_op_out=Belong(0),
                                                                           data_dictionary_out=expected)
        print_and_log("Test Case 9 Passed: expected ValueError, got ValueError")

        # Caso 10
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 4, 1, 1], 'B': [2, 4, 12, 12, 12], 'C': [10, 0, 3, 2, 2], 'D': [8, 8, 8, 2, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Caso 11
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 4, 12, 12], 'B': [2, 0, 3, 12, 12], 'C': [10, 0, 8, 1, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                           special_type_input=SpecialType(1),
                                                                           derived_type_output=DerivedType(2),
                                                                           missing_values=missing_values,
                                                                           axis_param=None, belong_op_in=Belong(0),
                                                                           field_in=None, field_out=None,
                                                                           belong_op_out=Belong(0),
                                                                           data_dictionary_out=expected)
        print_and_log("Test Case 12 Passed: expected ValueError, got ValueError")

        # Caso 13
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 3, 3], 'C': [3, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=None, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Caso 14
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                           special_type_input=SpecialType(2),
                                                                           derived_type_output=DerivedType(1),
                                                                           missing_values=missing_values,
                                                                           axis_param=None, belong_op_in=Belong(0),
                                                                           field_in=None, field_out=None,
                                                                           belong_op_out=Belong(0),
                                                                           data_dictionary_out=expected)
        print_and_log("Test Case 14 Passed: expected ValueError, got ValueError")

        # Caso 15
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [3, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 1, 2], 'C': [0, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # Caso 17
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 17 Failed: Expected True, but got False"
        print_and_log("Test Case 17 Passed: Expected True, got True")

        # Caso 18
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 4, 1], 'C': [2, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Caso 19
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 19 Failed: Expected True, but got False"
        print_and_log("Test Case 19 Passed: Expected True, got True")

        # Caso 20
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 3, 2], 'C': [1, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 20 Failed: Expected True, but got False"
        print_and_log("Test Case 20 Passed: Expected True, got True")

        # Caso 21
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'T'
        field_out = 'T'
        # Aplicar la transformación de datos
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                           special_type_input=SpecialType(1),
                                                                           derived_type_output=DerivedType(2),
                                                                           missing_values=missing_values,
                                                                           axis_param=None, field_in=field_in,
                                                                           field_out=field_out, belong_op_in=Belong(0),
                                                                           belong_op_out=Belong(0),
                                                                           data_dictionary_out=expected)
        print_and_log("Test Case 21 Passed: expected ValueError, got ValueError")

        # Caso 22
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 0, 0, 0], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 0, 0, 0], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       field_in=field_in, field_out=field_out,
                                                                       belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 23 Failed: Expected True, but got False"
        print_and_log("Test Case 23 Passed: Expected True, got True")

        # Caso 24
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = None
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(1),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 24 Failed: Expected True, but got False"
        print_and_log("Test Case 24 Passed: Expected True, got True")

        # Caso 25
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = None
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 25 Failed: Expected True, but got False"
        print_and_log("Test Case 25 Passed: Expected True, got True")

        # Caso 26
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        field_in = 'A'
        field_out = 'A'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 4, 1, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 26 Failed: Expected True, but got False"
        print_and_log("Test Case 26 Passed: Expected True, got True")

        # Caso 27
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [3, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 27 Failed: Expected True, but got False"
        print_and_log("Test Case 27 Passed: Expected True, got True")

        # Caso 28
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 29
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=1, field_in=field_in,
                                                                       field_out=field_out, belong_op_in=Belong(0),
                                                                       belong_op_out=Belong(0),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 29 Failed: Expected True, but got False"
        print_and_log("Test Case 29 Passed: Expected True, got True")

        # Caso 30
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 0, 0, 0, 0], 'B': [2, 12, 12, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected True, got True")

        # Caso 31
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 31 Failed: Expected True, but got False"
        print_and_log("Test Case 31 Passed: Expected True, got True")

        # Caso 32
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, 3, 3, 4, 2], 'B': [2, 3, 3, 12, 12], 'C': [10, 0, 3, 4, 2], 'D': [0, 8, 8, 4, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)
        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 32 Failed: Expected False, but got True"
        print_and_log("Test Case 32 Passed: Expected True, got True")


        # Caso 35
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 4, 2]})
        expected = expected.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(0),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 35 Failed: Expected True, but got False"
        print_and_log("Test Case 35 Passed: Expected True, got True")

        # Caso 36
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 4, 3, 2], 'B': [2, 99, 4, 12, 12], 'C': [10, 0, 8, 76, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'C': 'float64',  # Convertir C a float64
            'D': 'float64'  # Convertir D a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(1),
                                                                       derived_type_output=DerivedType(1),
                                                                       missing_values=missing_values,
                                                                       axis_param=1, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 36 Failed: Expected True, but got False"
        print_and_log("Test Case 36 Passed: Expected True, got True")

        # Caso 37
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        # Definir la lista de valores invalidos
        missing_values = [1, 3, 4]
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 8, 8, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(0),
                                                                       derived_type_output=DerivedType(2),
                                                                       missing_values=missing_values,
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 37 Failed: Expected True, but got False"
        print_and_log("Test Case 37 Passed: Expected True, got True")


        # Caso 39
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 8, 12], 'C': [11, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64',  # Convertir A a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       axis_param=None, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 39 Failed: Expected True, but got False"
        print_and_log("Test Case 39 Passed: Expected True, got True")


        # Caso 41
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 8, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 41 Failed: Expected True, but got False"
        print_and_log("Test Case 41 Passed: Expected True, got True")

        # Caso 42
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [0, 3, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(2),
                                                                       axis_param=0, belong_op_in=Belong(0),
                                                                       field_in=None, field_out=None,
                                                                       belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 42 Failed: Expected False, but got True"
        print_and_log("Test Case 42 Passed: Expected True, got True")

        # Caso 43
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 2, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(0),
                                                                       field_in=field_in, field_out=field_out,
                                                                       belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 43 Failed: Expected True, but got False"
        print_and_log("Test Case 43 Passed: Expected True, got True")

        # Caso 44
        # Crear un DataFrame de prueba
        datadic = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 0, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        field_in = 'C'
        field_out = 'C'
        # Definir el resultado esperado
        expected = pd.DataFrame(
            {'A': [0, None, 3, 4, 1], 'B': [2, 3, 4, 12, 12], 'C': [10, 10, 3, 3, 2], 'D': [1, 3, 2, 1, 2]})
        expected = expected.astype({
            'A': 'float64'  # Convertir A a float64
        })
        # Aplicar la transformación de datos
        result = self.invariants.check_inv_special_value_derived_value(data_dictionary_in=datadic.copy(),
                                                                       special_type_input=SpecialType(2),
                                                                       derived_type_output=DerivedType(1),
                                                                       field_in=field_in, field_out=field_out,
                                                                       belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                       data_dictionary_out=expected)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 44 Failed: Expected False, but got True"
        print_and_log("Test Case 44 Passed: Expected True, got True")


        print_and_log("")
        print_and_log("-----------------------------------------------------------")
        print_and_log("")

    def execute_checkInv_SpecialValue_NumOp(self):
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

        # -----------------------------------BELONG-BELONG--------------------------------------
        # MISSING
        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 2.0, 2], 'B': [2, 3, 5, 6, 12], 'C': [10, 8, 5, 2, 0],
             'D': [8.2, 8.2, 6, 4, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]

        expected_df = pd.DataFrame(
            {'A': [0, 2, datadic['A'].mean(), datadic['A'].mean(), 1], 'B': [2, 3, 4, 6, 12],
             'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=None,
                                                                field_in='A', field_out='A')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 2 Failed: Expected True, but got False"
        print_and_log("Test Case 2 Passed: Expected True, got True")

        # Caso 3
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 3.5, 6, 3.5], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(2),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=1,
                                                                field_in='B', field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 3, 3, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        missing_values = [1, 3, 4]

        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 2, 2, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                    data_dictionary_out=expected_df,
                                                                    special_type_input=SpecialType(0),
                                                                    num_op_output=Operation(3),
                                                                    belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                    missing_values=missing_values, axis_param=0,
                                                                    field_in='C', field_out='C')
        print_and_log("Test Case 4 Passed: Expected ValueError, got ValueError")

        # Caso 5
        # Probamos a aplicar la operación closest sobre un dataframe correcto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in='A', field_out='A')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Invalid
        # Caso 6
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [3, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [8.2, 8.2, 6, 4, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in='D', field_out='D')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, datadic.mean().mean().round(0), datadic.mean().mean().round(0), 1], 'B': [2, datadic.mean().mean().round(0), datadic.mean().mean().round(0), 6, 12],
             'C': [10, 1, datadic.mean().mean().round(0), 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=None,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 9
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 0], 'B': [2, 2, 3, 6, 12], 'C': [10, 3, 4, np.NaN, 0], 'D': [2, 8.2, 2, 2, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")

        # Outliers
        # Caso 11
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [1, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64',  # Convertir D a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=None, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Caso 12
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        datadic = datadic.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [3.7, 1, 3, 3, 0],
             'D': [1, 8.2, 6, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=None, axis_param=None,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Caso 14
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 6, 3, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=None, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 14 Failed: Expected True, but got False"
        print_and_log("Test Case 14 Passed: Expected True, got True")

        # Caso 15
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 3.5, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=None, axis_param=0,
                                                                field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Caso 16
        # Probamos a aplicar la operación mean sobre un field concreto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 2.84, 2, 1.5, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        missing_values = [8.2]
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 16 Failed: Expected True, but got False"
        print_and_log("Test Case 16 Passed: Expected True, got True")

        # -----------------------------------BELONG-NOTBELONG--------------------------------------
        # MISSING
        # Caso 17
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 2.0, 2], 'B': [2, 3.33333333, 4.66666667, 6, 12], 'C': [10, 7.5, 5, 2.5, 0],
             'D': [8.2, 8.2, 6, 4, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 17 Failed: Expected False, but got True"
        print_and_log("Test Case 17 Passed: Expected False, got False")

        # Caso 18
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]

        expected_df = pd.DataFrame(
            {'A': [0, 2, datadic['A'].mean(), datadic['A'].mean(), 1], 'B': [2, 3, 4, 6, 12],
             'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=None,
                                                                field_in='A', field_out='A')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 18 Failed: Expected False, but got True"
        print_and_log("Test Case 18 Passed: Expected False, got False")

        # Caso 19
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 3.5, 6, 3.5], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(2),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=1,
                                                                field_in='B', field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

        # Caso 20
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 3, 3, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        missing_values = [1, 3, 4]

        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, None, 2, 2, 0], 'D': [1, 8.2, np.NaN, 1, 2]})
        expected_exception = ValueError
        with self.assertRaises(expected_exception) as context:
            result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                    data_dictionary_out=expected_df,
                                                                    special_type_input=SpecialType(0),
                                                                    num_op_output=Operation(3),
                                                                    belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                    missing_values=missing_values, axis_param=0,
                                                                    field_in='C', field_out='C')
        print_and_log("Test Case 20 Passed: Expected ValueError, got ValueError")

        # Caso 21
        # Probamos a aplicar la operación closest sobre un dataframe correcto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(0),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in='A', field_out='A')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 21 Failed: Expected False, but got True"
        print_and_log("Test Case 21 Passed: Expected False, got False")

        # Invalid
        # Caso 22
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [8.2, 8.2, 6, 4, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(0),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in='D', field_out='D')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 22 Failed: Expected True, but got False"
        print_and_log("Test Case 22 Passed: Expected True, got True")

        # Caso 23
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 8, 8, 8], 'B': [2, 8, 8, 6, 12],
             'C': [10, 6, 2, 2, 0], 'D': [2, 8.2, 6, 2, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 23 Failed: Expected True, but got False"
        print_and_log("Test Case 23 Passed: Expected True, got True")

        # Caso 25
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        missing_values = [1, 3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 0], 'B': [2, 2, 3, 6, 12], 'C': [10, 3, 4, np.NaN, 0], 'D': [2, 8.2, 2, 2, 2]})
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 25 Failed: Expected False, but got True"
        print_and_log("Test Case 25 Passed: Expected False, got False")

        # Caso 26
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        missing_values = [3, 4]
        expected_df = pd.DataFrame(
            {'A': [0, 2, 2, 3, 1], 'B': [2, 2, 3, 6, 12], 'C': [10, 6, 6, 6, 0], 'D': [1, 8.2, 2, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(1),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 26 Failed: Expected False, but got True"
        print_and_log("Test Case 26 Passed: Expected False, got False")

        # Outliers
        # Caso 27
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 6], 'C': [1, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64',  # Convertir D a float64
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(0),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=None, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 27 Failed: Expected False, but got True"
        print_and_log("Test Case 27 Passed: Expected False, got False")

        # Caso 28
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 1, 3, 3, 0], 'D': [1, 8.2, 6, 1, 2]})
        datadic = datadic.astype({
            'B': 'float64',  # Convertir B a float64
            'C': 'float64'  # Convertir C a float64
        })
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [5, 3, 4, 6, 7], 'C': [3.7, 1, 3, 3, 0],
             'D': [1, 8.2, 6, 1, 2]})

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=None, axis_param=None,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 28 Failed: Expected True, but got False"
        print_and_log("Test Case 28 Passed: Expected True, got True")

        # Caso 30
        # Probamos a aplicar la operación closest sobre un dataframe con missing values (existen valores nulos)
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 10], 'C': [10, 4, 3, np.NaN, 0], 'D': [1, 6, 3, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })

        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=None, axis_param=0,
                                                                field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 30 Failed: Expected False, but got True"
        print_and_log("Test Case 30 Passed: Expected False, got False")

        # Caso 31
        # Probamos a aplicar la operación closest sobre un dataframe sin nulos
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 3.5, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(3),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=None, axis_param=0,
                                                                field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 31 Failed: Expected False, but got True"
        print_and_log("Test Case 31 Passed: Expected False, got False")

        # Caso 32
        # Probamos a aplicar la operación mean sobre un field concreto
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 8.2, 2, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 4, 1], 'B': [2, 3, 4, 6, 12], 'C': [10, 6, 3, 3, 0], 'D': [1, 5, 2, 1, 2]})
        expected_df = expected_df.astype({
            'D': 'float64'  # Convertir D a float64
        })
        field_in = 'D'
        field_out = 'D'
        missing_values = [8.2]
        result = self.invariants.check_inv_special_value_num_op(data_dictionary_in=datadic.copy(),
                                                                data_dictionary_out=expected_df,
                                                                special_type_input=SpecialType(2),
                                                                num_op_output=Operation(1),
                                                                belong_op_in=Belong(0), belong_op_out=Belong(1),
                                                                missing_values=missing_values, axis_param=0,
                                                                field_in=field_in, field_out=field_out)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 32 Failed: Expected True, but got False"
        print_and_log("Test Case 32 Passed: Expected True, got True")

    def execute_checkInv_MissingValue_MissingValue(self):
        """
        Execute the simple tests of the function checkInv_MissingValue_MissingValue
        """

        # Caso 1
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(1), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Caso 2
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(0), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Caso 3
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, 5], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(0), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Caso 4
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 5, 1], 'B': [2, 3, 4, 6, 5], 'C': [5, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(0), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 4 Failed: Expected True, but got False"
        print_and_log("Test Case 4 Passed: Expected True, got True")

        # Caso 5
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 4, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(1), field_in=None, field_out=None)

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Caso 6
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(1), field_in='B', field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 6 Failed: Expected True, but got False"
        print_and_log("Test Case 6 Passed: Expected True, got True")

        # Caso 7
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(0), field_in='B', field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is False, "Test Case 7 Failed: Expected False, but got True"
        print_and_log("Test Case 7 Passed: Expected False, got False")

        # Caso 8
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, 5], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(0), field_in='B', field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Caso 9
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, 5, 1], 'B': [2, 3, 4, 6, 5], 'C': [5, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(0), field_in='B', field_out='B')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 9 Failed: Expected True, but got False"
        print_and_log("Test Case 9 Passed: Expected True, got True")

        # Caso 10
        datadic = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 3, 0], 'D': [1, 8.2, 3, 1, 2]})
        expected_df = pd.DataFrame(
            {'A': [0, 2, 3, np.NaN, 1], 'B': [2, 3, 4, 6, np.NaN], 'C': [np.NaN, 4, 3, 4, 0], 'D': [1, 8.2, 3, 1, 2]})

        result = self.invariants.check_inv_missing_value_missing_value(data_dictionary_in=datadic.copy(),
                                                                       data_dictionary_out=expected_df.copy(),
                                                                       belong_op_out=Belong(1), field_in='C', field_out='C')

        # Verificar si el resultado obtenido coincide con el esperado
        assert result is True, "Test Case 10 Failed: Expected True, but got False"
        print_and_log("Test Case 10 Passed: Expected True, got True")
