
# Importing libraries
import numpy as np
import pandas as pd

# Importing functions and classes from packages
from helpers.auxiliar import cast_type_FixValue, check_interval_condition
from helpers.invariant_aux import check_special_type_most_frequent, check_special_type_previous, check_special_type_next, \
    check_derived_type_col_row_outliers, check_special_type_median, check_special_type_interpolation, check_special_type_mean, \
    check_special_type_closest, check_interval_most_frequent, check_interval_previous, check_interval_next, \
    check_fix_value_most_frequent, check_fix_value_previous, check_fix_value_next, check_fix_value_interpolation, check_fix_value_mean, \
    check_fix_value_median, check_fix_value_closest, check_interval_interpolation, check_interval_mean, check_interval_median, \
    check_interval_closest
from helpers.logger import print_and_log
from helpers.transform_aux import get_outliers
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType, Belong


def check_inv_fix_value_fix_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                  input_values_list: list = None, output_values_list: list = None,
                                  belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                  data_type_input_list: list = None, data_type_output_list: list = None,
                                  field_in: str = None, field_out: str = None) -> bool:
    """
    Check the invariant of the FixValue - FixValue relation (Mapping) is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        data_dictionary_in: dataframe with the input data
        data_dictionary_out: dataframe with the output data
        data_type_input: data type of the input value
        fix_value_input: input value to check
        belong_op: condition to check the invariant
        data_type_output: data type of the output value
        fix_value_output: output value to check
        field_in: field to check the invariant
        field_out: field to check the invariant

    returns:
        True if the invariant is satisfied, False otherwise
    """
    if input_values_list.__sizeof__() != output_values_list.__sizeof__():
        raise ValueError("The input and output values lists must have the same length")

    if data_type_input_list.__sizeof__() != data_type_output_list.__sizeof__():
        raise ValueError("The input and output data types lists must have the same length")

    for i in range(len(input_values_list)):
        if data_type_input_list is not None and data_type_output_list is not None:  # If the data types are specified, the transformation is performed
            # Auxiliary function that changes the values of fix_value_input and fix_value_output to the data type in data_type_input and data_type_output respectively
            input_values_list[i], output_values_list[i] = cast_type_FixValue(data_type_input=data_type_input_list[i],
                                                                            fix_value_input=input_values_list[i],
                                                                            data_type_output=data_type_output_list[i],
                                                                            fix_value_output=output_values_list[i])

    result=None
    if belong_op_out == Belong.BELONG:
        result=True
    elif belong_op_out == Belong.NOTBELONG:
        result=False

    # Create a dictionary to store the mapping equivalence between the input and output values
    mapping_values = {}

    for input_value in input_values_list:
        if input_value not in mapping_values:
            mapping_values[input_value] = output_values_list[input_values_list.index(input_value)]

    if field_in is None:
        # Iterar sobre las filas y columnas de data_dictionary_in
        for column_index, column_name in enumerate(data_dictionary_in.columns):
            for row_index, value in data_dictionary_in[column_name].items():
                # Comprobar si el valor es igual a fix_value_input
                if value in mapping_values:
                    if not pd.isna(data_dictionary_out.loc[row_index, column_name]) and (type(mapping_values[value])
                                                                                        == str or type(
                                mapping_values[value]) == object):
                        if data_dictionary_out.loc[row_index, column_name].strip() != mapping_values[value].strip():
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in row: {row_index} and column: {column_name} value should be: {mapping_values[value].strip()} but is: {data_dictionary_out.loc[row_index, column_name].strip()}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Row: {row_index} and column: {column_name} value should be: {mapping_values[value].strip()} but is: {data_dictionary_out.loc[row_index, column_name].strip()}")
                    else:
                        # Comprobar si el valor correspondiente en data_dictionary_out coincide con fix_value_output
                        if data_dictionary_out.loc[row_index, column_name] != mapping_values[value]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in row: {row_index} and column: {column_name} value should be: {mapping_values[value]} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Row: {row_index} and column: {column_name} value should be: {mapping_values[value]} but is: {data_dictionary_out.loc[row_index, column_name]}")

    elif field_in is not None:
        if field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            for row_index, value in data_dictionary_in[field_in].items():
                # Comprobar si el valor es igual a fix_value_input
                if value in mapping_values:
                    if not pd.isna(data_dictionary_out.loc[row_index, field_out]) and (type(mapping_values[value])
                                                                                          == str or type(
                                mapping_values[value]) == object):
                        if data_dictionary_out.loc[row_index, field_out].strip() != mapping_values[value].strip():
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in row: {row_index} and column: {field_out} value should be: {mapping_values[value].strip()} but is: {data_dictionary_out.loc[row_index, field_out].strip()}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Row: {row_index} and column: {field_out} value should be: {mapping_values[value].strip()} but is: {data_dictionary_out.loc[row_index, field_out].strip()}")
                    else:
                        # Comprobar si el valor correspondiente en data_dictionary_out coincide con fix_value_output
                        if data_dictionary_out.loc[row_index, field_in] != mapping_values[value]:
                            if belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in row: {row_index} and column: {field_out} value should be: {mapping_values[value]} but is: {data_dictionary_out.loc[row_index, field_in]}")
                            elif belong_op_out == Belong.NOTBELONG:
                                result = True
                                print_and_log(
                                    f"Row: {row_index} and column: {field_out} value should be: {mapping_values[value]} but is: {data_dictionary_out.loc[row_index, field_in]}")

        elif field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")


    return True if result else False


def check_inv_fix_value_derived_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                      fix_value_input, derived_type_output: DerivedType, belong_op_in: Belong = Belong.BELONG,
                                      belong_op_out: Belong = Belong.BELONG, data_type_input: DataType = None,
                                      axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    # By default, if all values are equally frequent, it is replaced by the first value.
    # Check if it should only be done for rows and columns or also for the entire dataframe.
    """
    Check the invariant of the FixValue - DerivedValue relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        data_dictionary_in: dataframe with the input data
        data_dictionary_out: dataframe with the output data
        data_type_input: data type of the input value
        fix_value_input: input value to check
        belong_op_in: if condition to check the invariant
        belong_op_out: then condition to check the invariant
        derived_type_output: derived type of the output value
        axis_param: axis to check the invariant - 0: column, None: dataframe
        field_in: field to check the invariant
        field_out: field to check the invariant

    returns:
        True if the invariant is satisfied, False otherwise
    """
    if data_type_input is not None:  # If the data types are specified, the transformation is performed
        # Auxiliary function that changes the values of fix_value_input to the data type in data_type_input
        fix_value_input, fix_value_output = cast_type_FixValue(data_type_input, fix_value_input, None,
                                                           None)

    result = True

    if derived_type_output == DerivedType.MOSTFREQUENT:
        result = check_fix_value_most_frequent(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                               fix_value_input=fix_value_input, belong_op_out=belong_op_out,
                                               axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif derived_type_output == DerivedType.PREVIOUS:
        result = check_fix_value_previous(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                          fix_value_input=fix_value_input, belong_op_out=belong_op_out,
                                          axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif derived_type_output == DerivedType.NEXT:
        result = check_fix_value_next(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                      fix_value_input=fix_value_input, belong_op_out=belong_op_out,
                                      axis_param=axis_param, field_in=field_in, field_out=field_out)

    return True if result else False


def check_inv_fix_value_num_op(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                               fix_value_input, num_op_output: Operation, belong_op_in: Belong = Belong.BELONG,
                               belong_op_out: Belong = Belong.BELONG, data_type_input: DataType = None,
                               axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check the invariant of the FixValue - NumOp relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
    params:
        data_dictionary_in: dataframe with the input data
        data_dictionary_out: dataframe with the output data
        data_type_input: data type of the input value
        fix_value_input: input value to check
        belong_op_in: if condition to check the invariant
        belong_op_out: then condition to check the invariant
        num_op_output: operation to check the invariant
        axis_param: axis to check the invariant
        field: field to check the invariant
    Returns:
        dataDictionary with the fix_value_input values replaced by the result of the operation num_op_output
    """

    if data_type_input is not None:  # If the data types are specified, the transformation is performed
        # Auxiliary function that changes the values of fix_value_input to the data type in data_type_input
        fix_value_input, fix_value_output = cast_type_FixValue(data_type_input, fix_value_input, None,
                                                           None)
    result = True

    if num_op_output == Operation.INTERPOLATION:
        result = check_fix_value_interpolation(data_dictionary_in=data_dictionary_in,
                                               data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                               belong_op_out=belong_op_out,
                                               axis_param=axis_param, field_in=field_in, field_out=field_out)

    elif num_op_output == Operation.MEAN:
        result = check_fix_value_mean(data_dictionary_in=data_dictionary_in,
                                      data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                      belong_op_out=belong_op_out,
                                      axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.MEDIAN:
        result = check_fix_value_median(data_dictionary_in=data_dictionary_in,
                                        data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                        belong_op_out=belong_op_out,
                                        axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.CLOSEST:
        result = check_fix_value_closest(data_dictionary_in=data_dictionary_in,
                                         data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                         belong_op_out=belong_op_out,
                                         axis_param=axis_param, field_in=field_in, field_out=field_out)

    return True if result else False


def check_inv_interval_fix_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                 left_margin: float, right_margin: float, closure_type: Closure, fix_value_output,
                                 belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                 data_type_output: DataType = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check the invariant of the Interval - FixValue relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param data_type_output: data type of the output value
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param fix_value_output: output value to check
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant

    returns:
        True if the invariant is satisfied, False otherwise
    """

    if data_type_output is not None:  # If it is specified, the transformation is performed
        vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if field_in is None:
        for column_index, column_name in enumerate(data_dictionary_in.columns):
            for row_index, value in data_dictionary_in[column_name].items():
                if check_interval_condition(value, left_margin, right_margin, closure_type):
                    if not pd.isna(data_dictionary_out.loc[row_index, field_out]) and (type(fix_value_output) == str
                                                                                       or type(fix_value_output) == object):
                        if data_dictionary_out.loc[row_index, column_name].strip() != fix_value_output.strip():
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in row: {row_index} and column: {column_name} value should be: {fix_value_output.strip()} but is: {data_dictionary_out.loc[row_index, column_name].strip()}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True
                    else:
                        if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                result = False
                                print_and_log(
                                    f"Error in row: {row_index} and column: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                result = True

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")
        if not np.issubdtype(data_dictionary_in[field_in].dtype, np.number):
            raise ValueError("The field is not numeric")

        for row_index, value in data_dictionary_in[field_in].items():
            if check_interval_condition(value, left_margin, right_margin, closure_type):
                if not pd.isna(data_dictionary_out.loc[row_index, field_out]) and (type(fix_value_output) == str or
                        type(
                        fix_value_output) == object):
                    if data_dictionary_out.loc[row_index, field_out].strip() != fix_value_output.strip():
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(
                                f"Error in row: {row_index} and column: {field_out} value should be: {fix_value_output.strip()} but is: {data_dictionary_out.loc[row_index, field_out].strip()}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True
                else:
                    if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                            result = False
                            print_and_log(
                                f"Error in row: {row_index} and column: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                            result = True

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_inv_interval_derived_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                     left_margin: float, right_margin: float,
                                     closure_type: Closure, derived_type_output: DerivedType,
                                     belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                     axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check the invariant of the Interval - DerivedValue relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param derived_type_output: derived type of the output value
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant

    returns:
        True if the invariant is satisfied, False otherwise
    """
    result = False

    if derived_type_output == DerivedType.MOSTFREQUENT:
        result = check_interval_most_frequent(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                              left_margin=left_margin, right_margin=right_margin,
                                              closure_type=closure_type, belong_op_out=belong_op_out,
                                              axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif derived_type_output == DerivedType.PREVIOUS:
        result = check_interval_previous(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                         left_margin=left_margin, right_margin=right_margin,
                                         closure_type=closure_type, belong_op_out=belong_op_out,
                                         axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif derived_type_output == DerivedType.NEXT:
        result = check_interval_next(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                     left_margin=left_margin, right_margin=right_margin,
                                     closure_type=closure_type, belong_op_out=belong_op_out,
                                     axis_param=axis_param, field_in=field_in, field_out=field_out)

    return True if result else False


def check_inv_interval_num_op(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                              left_margin: float, right_margin: float, closure_type: Closure, num_op_output: Operation,
                              belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                              axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check the invariant of the FixValue - NumOp relation
    If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param left_margin: left margin of the interval
        :param right_margin: right margin of the interval
        :param closure_type: closure type of the interval
        :param num_op_output: operation to check the invariant
        :param belong_op_in: operation to check the invariant
        :param belong_op_out: operation to check the invariant
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant

    returns:
        True if the invariant is satisfied, False otherwise
    """

    result = False

    if num_op_output == Operation.INTERPOLATION:
        result = check_interval_interpolation(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                              left_margin=left_margin, right_margin=right_margin,
                                              closure_type=closure_type, belong_op_in=belong_op_in,
                                              belong_op_out=belong_op_out, axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.MEAN:
        result = check_interval_mean(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                     left_margin=left_margin, right_margin=right_margin,
                                     closure_type=closure_type, belong_op_in=belong_op_in,
                                     belong_op_out=belong_op_out, axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.MEDIAN:
        result = check_interval_median(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                       left_margin=left_margin, right_margin=right_margin,
                                       closure_type=closure_type, belong_op_in=belong_op_in,
                                       belong_op_out=belong_op_out, axis_param=axis_param, field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.CLOSEST:
        result = check_interval_closest(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                        left_margin=left_margin, right_margin=right_margin,
                                        closure_type=closure_type, belong_op_in=belong_op_in,
                                        belong_op_out=belong_op_out, axis_param=axis_param, field_in=field_in, field_out=field_out)

    return True if result else False


def check_inv_special_value_fix_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                      special_type_input: SpecialType, fix_value_output,
                                      belong_op_in: Belong = Belong.BELONG,
                                      belong_op_out: Belong = Belong.BELONG, data_type_output: DataType = None,
                                      missing_values: list = None, axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check the invariant of the SpecialValue - FixValue relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        :param data_dictionary_in: input dataframe with the data
        :param data_dictionary_out: output dataframe with the data
        :param special_type_input: special type of the input value
        :param data_type_output: data type of the output value
        :param fix_value_output: output value to check
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant

    returns:
        True if the invariant is satisfied, False otherwise
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = True
    elif belong_op_out == Belong.NOTBELONG:
        result = False

    keep_no_trans_result = True

    if data_type_output is not None:  # If it is specified, the casting is performed
        vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

    if field_in is None:
        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
            if special_type_input == SpecialType.MISSING:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
            elif special_type_input == SpecialType.INVALID:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
            elif special_type_input == SpecialType.OUTLIER:
                threshold = 1.5
                if axis_param is None:
                    Q1 = data_dictionary_in.stack().quantile(0.25)
                    Q3 = data_dictionary_in.stack().quantile(0.75)
                    IQR = Q3 - Q1
                    # Define the lower and upper bounds
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    # Identify the outliers in the dataframe
                    numeric_values = data_dictionary_in.select_dtypes(include=[np.number])
                    for col in numeric_values.columns:
                        for idx in numeric_values.index:
                            value = numeric_values.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")
                elif axis_param == 0:
                    # Iterate over each numeric column
                    for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                        # Calculate the Q1, Q3, and IQR for each column
                        Q1 = data_dictionary_in[col].quantile(0.25)
                        Q3 = data_dictionary_in[col].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the column
                        for idx in data_dictionary_in.index:
                            value = data_dictionary_in.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")
                elif axis_param == 1:
                    # Iterate over each row
                    for idx in data_dictionary_in.index:
                        # Calculate the Q1, Q3, and IQR for each row
                        Q1 = data_dictionary_in.loc[idx].quantile(0.25)
                        Q3 = data_dictionary_in.loc[idx].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the row
                        for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                            value = data_dictionary_in.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")

        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
            if special_type_input == SpecialType.MISSING:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = True
                                print_and_log(f"Row: {row_index} and column: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
            elif special_type_input == SpecialType.INVALID:
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = True
                                print_and_log(f"Row: {row_index} and column: {column_name} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {data_dictionary_in.loc[row_index, column_name]} but is: {data_dictionary_out.loc[row_index, column_name]}")
            elif special_type_input == SpecialType.OUTLIER:
                threshold = 1.5
                if axis_param is None:
                    Q1 = data_dictionary_in.stack().quantile(0.25)
                    Q3 = data_dictionary_in.stack().quantile(0.75)
                    IQR = Q3 - Q1
                    # Define the lower and upper bounds
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    # Identify the outliers in the dataframe
                    numeric_values = data_dictionary_in.select_dtypes(include=[np.number])
                    for col in numeric_values.columns:
                        for idx in numeric_values.index:
                            value = numeric_values.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")
                elif axis_param == 0:
                    # Iterate over each numeric column
                    for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                        # Calculate the Q1, Q3, and IQR for each column
                        Q1 = data_dictionary_in[col].quantile(0.25)
                        Q3 = data_dictionary_in[col].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the column
                        for idx in data_dictionary_in.index:
                            value = data_dictionary_in.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")
                elif axis_param == 1:
                    # Iterate over each row
                    for idx in data_dictionary_in.index:
                        # Calculate the Q1, Q3, and IQR for each row
                        Q1 = data_dictionary_in.loc[idx].quantile(0.25)
                        Q3 = data_dictionary_in.loc[idx].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the row
                        for col in data_dictionary_in.select_dtypes(include=[np.number]).columns:
                            value = data_dictionary_in.loc[idx, col]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, col] != fix_value_output:
                                    result = True
                                    print_and_log(f"Row: {idx} and column: {col} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, col]}")
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                    keep_no_trans_result = False
                                    print_and_log(f"Error in row: {idx} and column: {col} value should be: {value} but is: {data_dictionary_out.loc[idx, col]}")

    elif field_in is not None:
        if field_in in data_dictionary_in.columns and field_out in data_dictionary_out.columns:
            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                if special_type_input == SpecialType.MISSING:
                    for row_index, value in data_dictionary_in[field_in].items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                                result = False
                                print_and_log(f"Error in row: {row_index} and column: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_in]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field_out] != value:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_in]}")

                elif special_type_input == SpecialType.INVALID:
                    for row_index, value in data_dictionary_in[field_in].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                                result = False
                                print_and_log(f"Error in row: {row_index} and column: {field_out} value should not be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field_out] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field_out])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                elif special_type_input == SpecialType.OUTLIER:
                    threshold = 1.5
                    # Calculate the Q1, Q3, and IQR for each column
                    Q1 = data_dictionary_in[field_in].quantile(0.25)
                    Q3 = data_dictionary_in[field_in].quantile(0.75)
                    IQR = Q3 - Q1
                    # Define the lower and upper bounds
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    # Identify the outliers in the column
                    for idx in data_dictionary_in.index:
                        value = data_dictionary_in.loc[idx, field_in]
                        is_outlier = (value < lower_bound) or (value > upper_bound)
                        if is_outlier:
                            if data_dictionary_out.loc[idx, field_out] != fix_value_output:
                                result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should not be: {fix_value_output} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[idx, field_out] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, field_out])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {value} but is: {data_dictionary_out.loc[idx, field_out]}")

            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                if special_type_input == SpecialType.MISSING:
                    for row_index, value in data_dictionary_in[field_in].items():
                        if value in missing_values or pd.isnull(value):
                            if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                                result = True
                                print_and_log(f"Row: {row_index} and column: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field_out] != value:
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                elif special_type_input == SpecialType.INVALID:
                    for row_index, value in data_dictionary_in[field_in].items():
                        if value in missing_values:
                            if data_dictionary_out.loc[row_index, field_out] != fix_value_output:
                                result = True
                                print_and_log(f"Row: {row_index} and column: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[row_index, field_out]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field_out] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field_out])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {row_index} and column: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                elif special_type_input == SpecialType.OUTLIER:
                    threshold = 1.5
                    # Calculate the Q1, Q3, and IQR for each column
                    Q1 = data_dictionary_in[field_in].quantile(0.25)
                    Q3 = data_dictionary_in[field_in].quantile(0.75)
                    IQR = Q3 - Q1
                    # Define the lower and upper bounds
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    # Identify the outliers in the column
                    for idx in data_dictionary_in.index:
                        value = data_dictionary_in.loc[idx, field_in]
                        is_outlier = (value < lower_bound) or (value > upper_bound)
                        if is_outlier:
                            if data_dictionary_out.loc[idx, field_out] != fix_value_output:
                                result = True
                                print_and_log(f"Row: {idx} and column: {field_out} value should be: {fix_value_output} but is: {data_dictionary_out.loc[idx, field_out]}")
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[idx, field_out] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, field_out])):
                                keep_no_trans_result = False
                                print_and_log(f"Error in row: {idx} and column: {field_out} value should be: {value} but is: {data_dictionary_out.loc[idx, field_out]}")

        elif field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")

    # Checks that the not transformed cells are not modified
    if keep_no_trans_result == False:
        return False
    else:
        return True if result else False


def check_inv_special_value_derived_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                          special_type_input: SpecialType, derived_type_output: DerivedType,
                                          belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                          missing_values: list = None, axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check the invariant of the SpecialValue - DerivedValue relation
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param special_type_input: special type of the input value
        :param derived_type_output: derived type of the output value
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant

    returns:
        True if the invariant is satisfied, False otherwise
    """
    result = True

    if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
        if derived_type_output == DerivedType.MOSTFREQUENT:
            result = check_special_type_most_frequent(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                                      special_type_input=special_type_input, belong_op_out=belong_op_out,
                                                      missing_values=missing_values, axis_param=axis_param, field_in=field_in, field_out=field_out)
        elif derived_type_output == DerivedType.PREVIOUS:
            result = check_special_type_previous(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                                 special_type_input=special_type_input,
                                                 belong_op_out=belong_op_out, missing_values=missing_values,
                                                 axis_param=axis_param, field_in=field_in, field_out=field_out)
        elif derived_type_output == DerivedType.NEXT:
            result = check_special_type_next(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                             special_type_input=special_type_input, belong_op_out=belong_op_out,
                                             missing_values=missing_values, axis_param=axis_param, field_in=field_in, field_out=field_out)

    elif special_type_input == SpecialType.OUTLIER:
        data_dictionary_outliers_mask = get_outliers(data_dictionary_in, field_in, axis_param)

        if axis_param is None:
            missing_values = data_dictionary_in.where(data_dictionary_outliers_mask == 1).stack().tolist()
            if derived_type_output == DerivedType.MOSTFREQUENT:
                result = check_special_type_most_frequent(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                                          special_type_input=special_type_input, belong_op_out=belong_op_out,
                                                          missing_values=missing_values, axis_param=axis_param, field_in=field_in, field_out=field_out)
            elif derived_type_output == DerivedType.PREVIOUS:
                result = check_special_type_previous(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                                     special_type_input=special_type_input,
                                                     belong_op_out=belong_op_out, missing_values=missing_values,
                                                     axis_param=axis_param, field_in=field_in, field_out=field_out)
            elif derived_type_output == DerivedType.NEXT:
                result = check_special_type_next(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                                 special_type_input=special_type_input, belong_op_out=belong_op_out,
                                                 missing_values=missing_values, axis_param=axis_param, field_in=field_in, field_out=field_out)

        elif axis_param == 0 or axis_param == 1:
            result = check_derived_type_col_row_outliers(derivedTypeOutput=derived_type_output, data_dictionary_in=data_dictionary_in,
                                                         data_dictionary_out=data_dictionary_out,
                                                         outliers_dataframe_mask=data_dictionary_outliers_mask,
                                                         belong_op_in=belong_op_in, belong_op_out=belong_op_out,
                                                         axis_param=axis_param, field_in=field_in, field_out=field_out)

    return True if result else False


def check_inv_special_value_num_op(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                   special_type_input: SpecialType, num_op_output: Operation,
                                   belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                   missing_values: list = None, axis_param: int = None, field_in: str = None, field_out: str = None) -> bool:
    """
    Check the invariant of the SpecialValue - NumOp relation is satisfied in the dataDicionary_out
    respect to the data_dictionary_in
    params:
        :param data_dictionary_in: dataframe with the data
        :param data_dictionary_out: dataframe with the data
        :param special_type_input: special type of the input value
        :param num_op_output: operation to check the invariant
        :param belong_op_in: if condition to check the invariant
        :param belong_op_out: then condition to check the invariant
        :param missing_values: list of missing values
        :param axis_param: axis to check the invariant
        :param field_in: field to check the invariant
        :param field_out: field to check the invariant

    returns:
        True if the invariant is satisfied, False otherwise
    """

    data_dictionary_outliers_mask = None
    result = True

    if special_type_input == SpecialType.OUTLIER:
        data_dictionary_outliers_mask = get_outliers(data_dictionary_in, field_in, axis_param)

    if num_op_output == Operation.INTERPOLATION:
        result = check_special_type_interpolation(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                                  special_type_input=special_type_input, belong_op_in=belong_op_in,
                                                  belong_op_out=belong_op_out,
                                                  data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                                  missing_values=missing_values, axis_param=axis_param,
                                                  field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.MEAN:
        result = check_special_type_mean(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                         special_type_input=special_type_input, belong_op_in=belong_op_in,
                                         belong_op_out=belong_op_out,
                                         data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                         missing_values=missing_values, axis_param=axis_param,
                                         field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.MEDIAN:
        result = check_special_type_median(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                           special_type_input=special_type_input, belong_op_in=belong_op_in,
                                           belong_op_out=belong_op_out,
                                           data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                           missing_values=missing_values, axis_param=axis_param,
                                           field_in=field_in, field_out=field_out)
    elif num_op_output == Operation.CLOSEST:
        result = check_special_type_closest(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                            special_type_input=special_type_input, belong_op_in=belong_op_in,
                                            belong_op_out=belong_op_out,
                                            data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                            missing_values=missing_values, axis_param=axis_param,
                                            field_in=field_in, field_out=field_out)

    return True if result else False


def check_inv_missing_value_missing_value(data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                          belong_op_out: Belong = Belong.BELONG, field_in: str = None, field_out: str = None) -> bool:
    """
    This function checks if the invariant of the MissingValue - MissingValue relation is satisfied in the output dataframe
    with respect to the input dataframe. The invariant is satisfied if all missing values in the input dataframe are still
    missing in the output dataframe.

    Parameters:
        data_dictionary_in (pd.DataFrame): The input dataframe.
        data_dictionary_out (pd.DataFrame): The output dataframe.
        belong_op_out (Belong): The condition to check the invariant. If it's Belong.BELONG, the function checks if the missing values are still missing.
                                If it's Belong.NOTBELONG, the function checks if the missing values are not missing anymore.
        field_in (str): The specific field (column) to check the invariant. If it's None, the function checks all fields.
        field_out (str): The specific field (column) to check the invariant. If it's None, the function checks all fields.

    Returns:
        bool: True if the invariant is satisfied, False otherwise.
    """
    result = None
    if belong_op_out == Belong.BELONG:
        result = False
    elif belong_op_out == Belong.NOTBELONG:
        result = True

    if field_in is None:
        for column_index, column_name in enumerate(data_dictionary_in.columns):
            for row_index, value in data_dictionary_in[column_name].items():
                if pd.isnull(value):
                    if not pd.isnull(data_dictionary_out.loc[row_index, column_name]):
                        if belong_op_out == Belong.BELONG:
                            result = True
                            print_and_log(f"Row: {row_index} and column: {column_name} value should be: {value} but is: {data_dictionary_out.loc[row_index, column_name]}")
                        elif belong_op_out == Belong.NOTBELONG:
                            result = False
                            print_and_log(f"Error in row: {row_index} and column: {column_name} value should be: {value} but is: {data_dictionary_out.loc[row_index, column_name]}")

    elif field_in is not None:
        if field_in not in data_dictionary_in.columns or field_out not in data_dictionary_out.columns:
            raise ValueError("The field does not exist in the dataframe")
        for row_index, value in data_dictionary_in[field_in].items():
            if pd.isnull(value):
                if not pd.isnull(data_dictionary_out.loc[row_index, field_out]):
                    if belong_op_out == Belong.BELONG:
                        result = True
                        print_and_log(f"Row: {row_index} and column: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")
                    elif belong_op_out == Belong.NOTBELONG:
                        result = False
                        print_and_log(f"Error in row: {row_index} and column: {field_out} value should be: {value} but is: {data_dictionary_out.loc[row_index, field_out]}")

    return True if result else False
