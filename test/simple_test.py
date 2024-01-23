import pandas as pd

from functions.contract_pre_post import ContractsPrePost
from helpers.enumerations import Belong, Operator
from helpers.logger import print_and_log


class ContractTest:
    def __init__(self):
        self.pre_post = ContractsPrePost()

    def execute_CheckFieldRange_Tests(self):
        print_and_log("Testing CheckFieldRange Function")
        print_and_log("")

        print_and_log("Casos Básicos solitados en la especificación del contrato:")

        # Case 1 of checkFieldRange
        # Check that fields 'c1' and 'c2' belong to the data dictionary. It must return True
        fields = ['c1', 'c2']
        data_dictionary = pd.DataFrame(columns=['c1', 'c2'])
        belong = 0
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Case 2 of checkFieldRange
        # Check that fields 'c2' and 'c3' belong to the data dictionary. It must return False as 'c3' does not belong
        fields = ['c2', 'c3']
        data_dictionary = pd.DataFrame(columns=['c1', 'c2'])
        belong = 0
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Case 3 of checkFieldRange
        # Check that fields 'c2' and 'c3' don't belong to the data dictionary.It must return True as 'c3' doesn't belong
        fields = ['c2', 'c3']
        data_dictionary = pd.DataFrame(columns=['c1', 'c2'])
        belong = 1
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        print_and_log("")
        print_and_log("Casos Básicos añadidos:")

        # Case 4 of checkFieldRange
        # Check that fields 'c1' and 'c2' don't belong to the data dictionary. It must return False as both belong
        fields = ['c1', 'c2']
        data_dictionary = pd.DataFrame(columns=['c1', 'c2'])
        belong = 1
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Case 5 of checkFieldRange
        # Check that fields 'c2' and 'c1' belong to the data dictionary. It must return True as both belong
        fields = ['c2', 'c1']
        data_dictionary = pd.DataFrame(columns=['c1', 'c2'])
        belong = 0
        result = self.pre_post.checkFieldRange(fields=fields, dataDictionary=data_dictionary, belongOp=Belong(belong))
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")
        print_and_log("")

    def execute_CheckFixValueRangeString_Tests(self):
        print_and_log("Testing CheckFixValueRangeString Function")

        print_and_log("")
        print_and_log("Casos Básicos solitados en la especificación del contrato:")

        # Example 13 of checkFixValueRangeString
        # Check that value None belongs to the data dictionary in field 'c1' and that
        # it appears less or equal than 30% of the times
        value = None
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field = 'c1'
        quant_op = 2  # lessEqual
        quant_rel = 0.3
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                        belongOp=Belong(belongOp), field=field, quant_rel=quant_rel,
                                                        quant_op=Operator(quant_op))
        assert result is True, "Test Case 13 Failed: Expected True, but got False"
        print_and_log("Test Case 13 Passed: Expected True, got True")

        # Example 14 of checkFixValueRangeString
        # Check that value None belongs to the data dictionary in field 'c1' and that
        # it appears less or equal than 30% of the times
        value = None
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', None, None, None, None]})
        belongOp = 0  # Belong
        field = 'c1'
        quant_op = 2  # lessEqual
        quant_rel = 0.3
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                        belongOp=Belong(belongOp), field=field, quant_rel=quant_rel,
                                                        quant_op=Operator(quant_op))
        assert result is False, "Test Case 14 Failed: Expected False, but got True"
        print_and_log("Test Case 14 Passed: Expected False, got False")

        # Example 18 of checkFixValueRangeString
        # Check that value 1 doesn't belong to the data dictionary in field 'c1'
        value = '1'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 1  # NotBelong
        field = 'c1'
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                        belongOp=Belong(belongOp), field=field)
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print_and_log("")
        print_and_log("Casos añadidos:")

        # Example 1 of checkFixValueRangeString
        value = '3'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '3', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field = None  # None
        quant_op = None  # None
        quant_rel = 0.3
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                   belongOp=Belong(belongOp), field=field, quant_rel=quant_rel,
                                                   quant_op=quant_op)
        assert result is True, "Test Case 1 Failed: Expected True, but got False"
        print_and_log("Test Case 1 Passed: Expected True, got True")

        # Example 2 of checkFixValueRangeString
        value = '3'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field = None  # None
        quant_op = None  # None
        quant_rel = 0.3
        # Ejecutar la función y verificar que devuelve False
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                   belongOp=Belong(belongOp), field=field, quant_rel=quant_rel,
                                                   quant_op=quant_op)
        assert result is False, "Test Case 2 Failed: Expected False, but got True"
        print_and_log("Test Case 2 Passed: Expected False, got False")

        # Example 3 of checkFixValueRangeString
        value = '0'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_rel = 0.1

        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                   belongOp=Belong(belongOp), field=field, quant_rel=quant_rel,
                                                   quant_op=Operator(quant_op))
        assert result is True, "Test Case 3 Failed: Expected True, but got False"
        print_and_log("Test Case 3 Passed: Expected True, got True")

        # Example 4 of checkFixValueRangeString
        value = '0'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_rel = 0.7

        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                   belongOp=Belong(belongOp), field=field, quant_rel=quant_rel,
                                                   quant_op=Operator(quant_op))
        assert result is False, "Test Case 4 Failed: Expected False, but got True"
        print_and_log("Test Case 4 Passed: Expected False, got False")

        # Example 5 of checkFixValueRangeString
        value = '0'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_abs = 3

        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                   belongOp=Belong(belongOp), field=field, quant_abs=quant_abs,
                                                   quant_op=Operator(quant_op))
        assert result is True, "Test Case 5 Failed: Expected True, but got False"
        print_and_log("Test Case 5 Passed: Expected True, got True")

        # Example 6 of checkFixValueRangeString
        value = None
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field = None
        quant_op = 1  # greater
        quant_abs = 5

        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                   belongOp=Belong(belongOp), field=field, quant_abs=quant_abs,
                                                   quant_op=Operator(quant_op))
        assert result is False, "Test Case 6 Failed: Expected False, but got True"
        print_and_log("Test Case 6 Passed: Expected False, got False")

        # Example 8 of checkFixValueRangeString
        value = '3'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 1  # Not Belong
        # # Ejecutar la función y verificar que devuelve False
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                        belongOp=Belong(belongOp))
        assert result is True, "Test Case 8 Failed: Expected True, but got False"
        print_and_log("Test Case 8 Passed: Expected True, got True")

        # Example 9 of checkFixValueRangeString
        value = '0'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 1  # Not Belong
        # # Ejecutar la función y verificar que devuelve False
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
                                                        belongOp=Belong(belongOp))
        assert result is False, "Test Case 9 Failed: Expected False, but got True"
        print_and_log("Test Case 9 Passed: Expected False, got False")

        # Example 11 of checkFixValueRangeString
        value = '0'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field= 'c1'
        # # Ejecutar la función y verificar que devuelve False
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,field=field,
                                                        belongOp=Belong(belongOp))
        assert result is True, "Test Case 11 Failed: Expected True, but got False"
        print_and_log("Test Case 11 Passed: Expected True, got True")

        # Example 12 of checkFixValueRangeString
        value = '5'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field= 'c1'
        # # Ejecutar la función y verificar que devuelve False
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,field=field,
                                                        belongOp=Belong(belongOp))
        assert result is False, "Test Case 12 Failed: Expected False, but got True"
        print_and_log("Test Case 12 Passed: Expected False, got False")

        # Example 15 of checkFixValueRangeString
        value = '0'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field= 'c1'
        quant_op = 1  # greater
        quant_abs = 3
        # # Ejecutar la función y verificar que devuelve False
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,field=field,
                                                        quant_abs=quant_abs, quant_op=Operator(quant_op), belongOp=Belong(belongOp))
        assert result is True, "Test Case 15 Failed: Expected True, but got False"
        print_and_log("Test Case 15 Passed: Expected True, got True")

        # Example 16 of checkFixValueRangeString
        value = '0'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 0  # Belong
        field= 'c1'
        quant_op = 1  # greater
        quant_abs = 10
        # # Ejecutar la función y verificar que devuelve False
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,field=field,
                                                        quant_abs=quant_abs, quant_op=Operator(quant_op), belongOp=Belong(belongOp))
        assert result is False, "Test Case 16 Failed: Expected False, but got True"
        print_and_log("Test Case 16 Passed: Expected False, got False")

        # Example 18 of checkFixValueRangeString
        value = '3'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 1  # Not Belong
        field= 'c1'
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,field=field,
                                                        belongOp=Belong(belongOp))
        assert result is True, "Test Case 18 Failed: Expected True, but got False"
        print_and_log("Test Case 18 Passed: Expected True, got True")

        # Example 19 of checkFixValueRangeString
        value = '0'
        dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        belongOp = 1  # Not Belong
        field= 'c1'
        result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,field=field,
                                                        belongOp=Belong(belongOp))
        assert result is False, "Test Case 19 Failed: Expected False, but got True"
        print_and_log("Test Case 19 Passed: Expected False, got False")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Casos de error añadidos
        #print_and_log("")
        #print_and_log("Casos de error añadidos:")

        # Example 4.5 of checkFixValueRangeString
        # CASO DE QUE quant_rel y quant_abs NO SEAN None A LA VEZ (existen los dos) VALUERROR
        #value = None
        #dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        #belongOp = 0  # Belong
        #field = None
        #quant_op = 2  # lessEqual
        #quant_rel = 0.4
        #quant_abs = 50
        #result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
        #                                                belongOp=Belong(belongOp), field=field, quant_rel=quant_rel,
        #                                                quant_abs=quant_abs, quant_op=Operator(quant_op))
        #assert result is ValueError and "Test Case 4.5 Failed: Expected ValueError, but got True/False"
        #print_and_log("Test Case 4.5 Passed: Expected ValueError, got ValueError")

        # Example 7 of checkFixValueRangeString
        # CASO DE QUE no existan ni quant_rel ni quant_abs cuando belongOp es BELONG Y quant_op no es None VALUERROR
        #value = '0'
        #dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        #belongOp = 0  # Belong
        #field = None
        #quant_op = 2  # lessEqual
        #result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
        #                                                belongOp=Belong(belongOp), field=field, quant_op=Operator(quant_op))
        #assert result is ValueError and "Test Case 7 Failed: Expected ValueError, but got True/False"
        #print_and_log("Test Case 7 Passed: Expected ValueError, got ValueError")

        # # Example 10 of checkFixValueRangeString
        # value = '0'
        # dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        # belongOp = 1  # Not Belong
        # quant_op = 3
        # # # Ejecutar la función y verificar que devuelve False
        # result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
        #                                                 belongOp=Belong(belongOp), quant_op=Operator(quant_op))
        # assert result is ValueError, "Test Case 10 Failed: Expected ValueError, but got True/False"
        # print_and_log("Test Case 10 Passed: Expected ValueError, got ValueError")

        # Example 14.5 of checkFixValueRangeString
        # CASO DE QUE NO SE PUEDEN PROPORCIONAR QUANT_REL Y QUANT_ABS A LA VEZ??? VALUERROR
        #value = None
        #dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        #belongOp = 0  # Belong
        #field = 'c1'
        #quant_op = 2  # lessEqual
        #quant_rel = 0.4
        #quant_abs = 50
        #result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
        #                                                belongOp=Belong(belongOp), field=field, quant_rel=quant_rel,
        #                                                quant_abs=quant_abs, quant_op=Operator(quant_op))
        #assert result is ValueError and "Test Case 14.5 Failed: Expected ValueError, but got True/False"
        #print_and_log("Test Case 14.5 Passed: Expected ValueError, got ValueError")

        # Example 17 of checkFixValueRangeString
        # CASO DE QUE no existan ni quant_rel ni quant_abs cuando belongOp es BELONG Y quant_op no es None VALUERROR
        #value = '0'
        #dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        #belongOp = 0  # Belong
        #field = 'c1'
        #quant_op = 2  # lessEqual
        #result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,
        #                                               belongOp=Belong(belongOp), field=field, quant_op=Operator(quant_op))
        #assert result is ValueError and "Test Case 17 Failed: Expected ValueError, but got True/False"
        #print_and_log("Test Case 17 Passed: Expected ValueError, got ValueError")

        # Example 20 of checkFixValueRangeString
        #value = '0'
        #dataDictionary = pd.DataFrame(data={'c1': ['0', '0', '0', '0', '0', '0', '0', None, None, None]})
        #belongOp = 1  # Not Belong
        #field = 'c1'
        #quant_op = 3
        #result = self.pre_post.checkFixValueRangeString(value=value, dataDictionary=dataDictionary,field=field,
        #                                                belongOp=Belong(belongOp), quant_op=Operator(quant_op))
        #assert result is ValueError, "Test Case 20 Failed: Expected ValueError, but got True/False"
        #print_and_log("Test Case 20 Passed: Expected ValueError, got ValueError")
