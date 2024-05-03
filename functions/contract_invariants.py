
# Importing libraries
import numpy as np
import pandas as pd

# Importing functions and classes from packages
from helpers.auxiliar import cast_type_FixValue, find_closest_value, check_interval_condition
from helpers.invariant_aux import check_special_type_most_frequent, check_special_type_previous, check_special_type_next, \
    check_derived_type_col_row_outliers, check_special_type_median, check_special_type_interpolation, check_special_type_mean, \
    check_special_type_closest, check_interval_most_frequent, check_interval_previous, check_interval_next, \
    check_fix_value_most_frequent, check_fix_value_previous, check_fix_value_next, check_fix_value_interpolation, check_fix_value_mean, \
    check_fix_value_median, check_fix_value_closest, check_interval_interpolation, check_interval_mean, check_interval_median, \
    check_interval_closest
from helpers.transform_aux import get_outliers
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType, Belong




class Invariants:
    # FixValue - FixValue, FixValue - DerivedValue, FixValue - NumOp
    # Interval - FixValue, Interval - DerivedValue, Interval - NumOp
    # SpecialValue - FixValue, SpecialValue - DerivedValue, SpecialValue - NumOp

    def check_inv_fix_value_fix_value(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                      fix_value_input, fix_value_output, belong_op_in: Belong = Belong.BELONG,
                                      belong_op_out: Belong = Belong.BELONG, data_type_input: DataType = None,
                                      data_type_output: DataType = None, field: str = None) -> bool:
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
            field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        if data_type_input is not None and data_type_output is not None:  # If the data types are specified, the transformation is performed
            # Auxiliary function that changes the values of fix_value_input and fix_value_output to the data type in data_type_input and data_type_output respectively
            fix_value_input, fix_value_output = cast_type_FixValue(data_type_input, fix_value_input, data_type_output,
                                                               fix_value_output)
        result=None
        if belong_op_out == Belong.BELONG:
            result=True
        elif belong_op_out == Belong.NOTBELONG:
            result=False

        if field is None:
            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                # Iterar sobre las filas y columnas de data_dictionary_in
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        # Comprobar si el valor es igual a fix_value_input
                        if value == fix_value_input:
                            # Comprobar si el valor correspondiente en data_dictionary_out coincide con fix_value_output
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = False
                                print("Error in row: ", row_index, " and column: ", column_name, " value should be: ", fix_value_output, " but is: ", data_dictionary_out.loc[row_index, column_name])
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                result = False
                                print("Error in row: ", row_index, " and column: ", column_name, " value should be: ", data_dictionary_in.loc[row_index, column_name], " but is: ", data_dictionary_out.loc[row_index, column_name])
            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                # Iterar sobre las filas y columnas de data_dictionary_in
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        # Comprobar si el valor es igual a fix_value_input
                        if value == fix_value_input:
                            # Comprobar si el valor correspondiente en data_dictionary_out coincide con fix_value_output
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                result = True
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                result = False
                                print("Error in row: ", row_index, " and column: ", column_name, " value should be: ", data_dictionary_in.loc[row_index, column_name], " but is: ", data_dictionary_out.loc[row_index, column_name])

        elif field is not None:
            if field in data_dictionary_in.columns and field in data_dictionary_out.columns:
                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                    for row_index, value in data_dictionary_in[field].items():
                        # Comprobar si el valor es igual a fix_value_input
                        if value == fix_value_input:
                            # Comprobar si el valor correspondiente en data_dictionary_out coincide con fix_value_output
                            if data_dictionary_out.loc[row_index, field] != fix_value_output:
                                result = False
                                print("Error in row: ", row_index, " and column: ", field, " value should be: ", fix_value_output, " but is: ", data_dictionary_out.loc[row_index, field])
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field])):
                                result = False
                                print("Error in row: ", row_index, " and column: ", field, " value should be: ", data_dictionary_in.loc[row_index, field], " but is: ", data_dictionary_out.loc[row_index, field])
                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                    for row_index, value in data_dictionary_in[field].items():
                        # Comprobar si el valor es igual a fix_value_input
                        if value == fix_value_input:
                            # Comprobar si el valor correspondiente en data_dictionary_out coincide con fix_value_output
                            if data_dictionary_out.loc[row_index, field] != fix_value_output:
                                result = True
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field])):
                                result = False
                                print("Error in row: ", row_index, " and column: ", field, " value should be: ", data_dictionary_in.loc[row_index, field], " but is: ", data_dictionary_out.loc[row_index, field])
            elif field not in data_dictionary_in.columns or field not in data_dictionary_out.columns:
                raise ValueError("The field does not exist in the dataframe")

        return True if result else False

    def check_inv_fix_value_derived_value(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                          fix_value_input, derived_type_output: DerivedType, belong_op_in: Belong = Belong.BELONG,
                                          belong_op_out: Belong = Belong.BELONG, data_type_input: DataType = None,
                                          axis_param: int = None, field: str = None) -> bool:
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
            field: field to check the invariant

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
                                                   fix_value_input=fix_value_input, belong_op_in=belong_op_in,
                                                   belong_op_out=belong_op_out, axis_param=axis_param, field=field)
        elif derived_type_output == DerivedType.PREVIOUS:
            result = check_fix_value_previous(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                              fix_value_input=fix_value_input, belong_op_in=belong_op_in,
                                              belong_op_out=belong_op_out, axis_param=axis_param, field=field)
        elif derived_type_output == DerivedType.NEXT:
            result = check_fix_value_next(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                          fix_value_input=fix_value_input, belong_op_in=belong_op_in,
                                          belong_op_out=belong_op_out, axis_param=axis_param, field=field)

        return True if result else False


    def check_inv_fix_value_num_op(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                   fix_value_input, num_op_output: Operation, belong_op_in: Belong = Belong.BELONG,
                                   belong_op_out: Belong = Belong.BELONG, data_type_input: DataType = None,
                                   axis_param: int = None, field: str = None) -> bool:
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
                                                   belong_op_in=belong_op_in, belong_op_out=belong_op_out,
                                                   axis_param=axis_param, field=field)

        elif num_op_output == Operation.MEAN:
            result = check_fix_value_mean(data_dictionary_in=data_dictionary_in,
                                          data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                          belong_op_in=belong_op_in, belong_op_out=belong_op_out,
                                          axis_param=axis_param, field=field)
        elif num_op_output == Operation.MEDIAN:
            result = check_fix_value_median(data_dictionary_in=data_dictionary_in,
                                            data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                            belong_op_in=belong_op_in, belong_op_out=belong_op_out,
                                            axis_param=axis_param, field=field)
        elif num_op_output == Operation.CLOSEST:
            result = check_fix_value_closest(data_dictionary_in=data_dictionary_in,
                                             data_dictionary_out=data_dictionary_out, fix_value_input=fix_value_input,
                                             belong_op_in=belong_op_in, belong_op_out=belong_op_out,
                                             axis_param=axis_param, field=field)

        return True if result else False

    def check_inv_interval_fix_value(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                     left_margin: float, right_margin: float, closure_type: Closure, fix_value_output,
                                     belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                     data_type_output: DataType = None, field: str = None) -> bool:
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
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """

        if data_type_output is not None:  # If it is specified, the transformation is performed
            vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

        if field is None:
            if (belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG) or (belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG):
                for column_index, column_name in enumerate(data_dictionary_in.columns):
                    for row_index, value in data_dictionary_in[column_name].items():
                        if check_interval_condition(value, left_margin, right_margin, closure_type):
                            if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                    return False
                                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                    return True
                        else: # Si el valor no es igual a fix_value_input
                            if data_dictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                return False

        elif field is not None:
            if field not in data_dictionary_in.columns or field not in data_dictionary_out.columns:
                raise ValueError("The field does not exist in the dataframe")
            if not np.issubdtype(data_dictionary_in[field].dtype, np.number):
                raise ValueError("The field is not numeric")

            if (belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG) or (belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG):
                for row_index, value in data_dictionary_in[field].items():
                    if check_interval_condition(value, left_margin, right_margin, closure_type):
                        if data_dictionary_out.loc[row_index, field] != fix_value_output:
                            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                                return False
                            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                                return True
                    else: # Si el valor no es igual a fix_value_input
                        if data_dictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field])):
                            return False


        if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
            return True
        elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
            return False
        else:
            return True

    def check_inv_interval_derived_value(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                         left_margin: float, right_margin: float,
                                         closure_type: Closure, derived_type_output: DerivedType,
                                         belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                         axis_param: int = None, field: str = None) -> bool:
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
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        result = True

        if derived_type_output == DerivedType.MOSTFREQUENT:
            result = check_interval_most_frequent(data_dictionary_in, data_dictionary_out, left_margin, right_margin,
                                                  closure_type, belong_op_in, belong_op_out, axis_param, field)
        elif derived_type_output == DerivedType.PREVIOUS:
            result = check_interval_previous(data_dictionary_in, data_dictionary_out, left_margin, right_margin,
                                             closure_type, belong_op_in, belong_op_out, axis_param, field)
        elif derived_type_output == DerivedType.NEXT:
            result = check_interval_next(data_dictionary_in, data_dictionary_out, left_margin, right_margin,
                                         closure_type, belong_op_in, belong_op_out, axis_param, field)

        return True if result else False

    def check_inv_interval_num_op(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                  left_margin: float, right_margin: float, closure_type: Closure, num_op_output: Operation,
                                  belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                  axis_param: int = None, field: str = None) -> bool:
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
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """

        result = True

        if num_op_output == Operation.INTERPOLATION:
            result = check_interval_interpolation(data_dictionary_in, data_dictionary_out, left_margin, right_margin,
                                                  closure_type, belong_op_in, belong_op_out, axis_param, field)
        elif num_op_output == Operation.MEAN:
            result = check_interval_mean(data_dictionary_in, data_dictionary_out, left_margin, right_margin,
                                         closure_type, belong_op_in, belong_op_out, axis_param, field)
        elif num_op_output == Operation.MEDIAN:
            result = check_interval_median(data_dictionary_in, data_dictionary_out, left_margin, right_margin,
                                           closure_type, belong_op_in, belong_op_out, axis_param, field)
        elif num_op_output == Operation.CLOSEST:
            result = check_interval_closest(data_dictionary_in, data_dictionary_out, left_margin, right_margin,
                                            closure_type, belong_op_in, belong_op_out, axis_param, field)

        return True if result else False


    def check_inv_special_value_fix_value(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                          special_type_input: SpecialType, fix_value_output,
                                          belong_op_in: Belong = Belong.BELONG,
                                          belong_op_out: Belong = Belong.BELONG, data_type_output: DataType = None,
                                          missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
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
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        if data_type_output is not None:  # If it is specified, the casting is performed
            vacio, fix_value_output = cast_type_FixValue(None, None, data_type_output, fix_value_output)

        if field is None:
            if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                if special_type_input == SpecialType.MISSING:
                    for column_index, column_name in enumerate(data_dictionary_in.columns):
                        for row_index, value in data_dictionary_in[column_name].items():
                            if value in missing_values or pd.isnull(value):
                                if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[row_index, column_name] != value:
                                    return False
                elif special_type_input == SpecialType.INVALID:
                    for column_index, column_name in enumerate(data_dictionary_in.columns):
                        for row_index, value in data_dictionary_in[column_name].items():
                            if value in missing_values:
                                if data_dictionary_out.loc[row_index, column_name] != fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                    return False
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
                                        return False
                                else: # Si el valor no es igual a fix_value_input
                                    if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                        return False
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
                                        return False
                                else: # Si el valor no es igual a fix_value_input
                                    if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                        return False
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
                                        return False
                                else: # Si el valor no es igual a fix_value_input
                                    if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                        return False

            elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                if special_type_input == SpecialType.MISSING:
                    for column_index, column_name in enumerate(data_dictionary_in.columns):
                        for row_index, value in data_dictionary_in[column_name].items():
                            if value in missing_values or pd.isnull(value):
                                if data_dictionary_out.loc[row_index, column_name] == fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[row_index, column_name] != value:
                                    return False
                elif special_type_input == SpecialType.INVALID:
                    for column_index, column_name in enumerate(data_dictionary_in.columns):
                        for row_index, value in data_dictionary_in[column_name].items():
                            if value in missing_values:
                                if data_dictionary_out.loc[row_index, column_name] == fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, column_name])):
                                    return False
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
                                    if data_dictionary_out.loc[idx, col] == fix_value_output:
                                        return False
                                else: # Si el valor no es igual a fix_value_input
                                    if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                        return False
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
                                    if data_dictionary_out.loc[idx, col] == fix_value_output:
                                        return False
                                else: # Si el valor no es igual a fix_value_input
                                    if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                        return False
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
                                    if data_dictionary_out.loc[idx, col] == fix_value_output:
                                        return False
                                else: # Si el valor no es igual a fix_value_input
                                    if data_dictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, col])):
                                        return False

        elif field is not None:
            if field in data_dictionary_in.columns and field in data_dictionary_out.columns:
                if belong_op_in == Belong.BELONG and belong_op_out == Belong.BELONG:
                    if special_type_input == SpecialType.MISSING:
                        for row_index, value in data_dictionary_in[field].items():
                            if value in missing_values or pd.isnull(value):
                                if data_dictionary_out.loc[row_index, field] != fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[row_index, field] != value:
                                    return False
                    elif special_type_input == SpecialType.INVALID:
                        for row_index, value in data_dictionary_in[field].items():
                            if value in missing_values:
                                if data_dictionary_out.loc[row_index, field] != fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field])):
                                    return False
                    elif special_type_input == SpecialType.OUTLIER:
                        threshold = 1.5
                        # Calculate the Q1, Q3, and IQR for each column
                        Q1 = data_dictionary_in[field].quantile(0.25)
                        Q3 = data_dictionary_in[field].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the column
                        for idx in data_dictionary_in.index:
                            value = data_dictionary_in.loc[idx, field]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, field] != fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, field] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, field])):
                                    return False

                elif belong_op_in == Belong.BELONG and belong_op_out == Belong.NOTBELONG:
                    if special_type_input == SpecialType.MISSING:
                        for row_index, value in data_dictionary_in[field].items():
                            if value in missing_values or pd.isnull(value):
                                if data_dictionary_out.loc[row_index, field] == fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[row_index, field] != value:
                                    return False
                    elif special_type_input == SpecialType.INVALID:
                        for row_index, value in data_dictionary_in[field].items():
                            if value in missing_values:
                                if data_dictionary_out.loc[row_index, field] == fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[row_index, field])):
                                    return False
                    elif special_type_input == SpecialType.OUTLIER:
                        threshold = 1.5
                        # Calculate the Q1, Q3, and IQR for each column
                        Q1 = data_dictionary_in[field].quantile(0.25)
                        Q3 = data_dictionary_in[field].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the column
                        for idx in data_dictionary_in.index:
                            value = data_dictionary_in.loc[idx, field]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if data_dictionary_out.loc[idx, field] == fix_value_output:
                                    return False
                            else: # Si el valor no es igual a fix_value_input
                                if data_dictionary_out.loc[idx, field] != value and not(pd.isnull(value) and pd.isnull(data_dictionary_out.loc[idx, field])):
                                    return False

            elif field not in data_dictionary_in.columns or field not in data_dictionary_out.columns:
                raise ValueError("The field does not exist in the dataframe")

        return True

    def check_inv_special_value_derived_value(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                              special_type_input: SpecialType, derived_type_output: DerivedType,
                                              belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                              missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
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
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        data_dictionary_outliers_mask = None
        result = True

        if special_type_input == SpecialType.OUTLIER:
            data_dictionary_outliers_mask = get_outliers(data_dictionary_in, field, axis_param)

            if axis_param is None:
                missing_values = data_dictionary_in.where(data_dictionary_outliers_mask == 1).stack().tolist()
        if special_type_input == SpecialType.MISSING or special_type_input == SpecialType.INVALID:
            if derived_type_output == DerivedType.MOSTFREQUENT:
                result = check_special_type_most_frequent(data_dictionary_in, data_dictionary_out, special_type_input,
                                                          belong_op_in, belong_op_out, missing_values, axis_param, field)
            elif derived_type_output == DerivedType.PREVIOUS:
                result = check_special_type_previous(data_dictionary_in, data_dictionary_out, special_type_input,
                                                     belong_op_in, belong_op_out, missing_values, axis_param, field)
            elif derived_type_output == DerivedType.NEXT:
                result = check_special_type_next(data_dictionary_in, data_dictionary_out, special_type_input,
                                                 belong_op_in, belong_op_out, missing_values, axis_param, field)

        elif special_type_input == SpecialType.OUTLIER:
            # IMPORTANT: The function getOutliers() does the same as apply_derivedTypeOutliers() but at the dataframe level.
            # If the outliers are applied at the dataframe level, previous and next cannot be applied.

            if axis_param is None:
                missing_values = data_dictionary_in.where(data_dictionary_outliers_mask == 1).stack().tolist()
                if derived_type_output == DerivedType.MOSTFREQUENT:
                    result = check_special_type_most_frequent(data_dictionary_in, data_dictionary_out, special_type_input,
                                                              belong_op_in, belong_op_out, missing_values, axis_param, field)
                elif derived_type_output == DerivedType.PREVIOUS:
                    result = check_special_type_previous(data_dictionary_in, data_dictionary_out, special_type_input,
                                                         belong_op_in, belong_op_out, missing_values, axis_param, field)
                elif derived_type_output == DerivedType.NEXT:
                    result = check_special_type_next(data_dictionary_in, data_dictionary_out, special_type_input,
                                                     belong_op_in, belong_op_out, missing_values, axis_param, field)

            elif axis_param == 0 or axis_param == 1:
                result = check_derived_type_col_row_outliers(derived_type_output, data_dictionary_in,
                                                             data_dictionary_out,
                                                             data_dictionary_outliers_mask, belong_op_in, belong_op_out,
                                                             axis_param, field)

        return True if result else False

    def check_inv_special_value_num_op(self, data_dictionary_in: pd.DataFrame, data_dictionary_out: pd.DataFrame,
                                       special_type_input: SpecialType, num_op_output: Operation,
                                       belong_op_in: Belong = Belong.BELONG, belong_op_out: Belong = Belong.BELONG,
                                       missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
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
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """

        data_dictionary_outliers_mask = None
        result = True

        if special_type_input == SpecialType.OUTLIER:
            data_dictionary_outliers_mask = get_outliers(data_dictionary_in, field, axis_param)

        if num_op_output == Operation.INTERPOLATION:
            result = check_special_type_interpolation(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                                      special_type_input=special_type_input, belong_op_in=belong_op_in,
                                                      belong_op_out=belong_op_out,
                                                      data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                                      missing_values=missing_values, axis_param=axis_param,
                                                      field=field)
        elif num_op_output == Operation.MEAN:
            result = check_special_type_mean(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                             special_type_input=special_type_input, belong_op_in=belong_op_in,
                                             belong_op_out=belong_op_out,
                                             data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                             missing_values=missing_values, axis_param=axis_param,
                                             field=field)
        elif num_op_output == Operation.MEDIAN:
            result = check_special_type_median(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                               special_type_input=special_type_input, belong_op_in=belong_op_in,
                                               belong_op_out=belong_op_out,
                                               data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                               missing_values=missing_values, axis_param=axis_param,
                                               field=field)
        elif num_op_output == Operation.CLOSEST:
            result = check_special_type_closest(data_dictionary_in=data_dictionary_in, data_dictionary_out=data_dictionary_out,
                                                special_type_input=special_type_input, belong_op_in=belong_op_in,
                                                belong_op_out=belong_op_out,
                                                data_dictionary_outliers_mask=data_dictionary_outliers_mask,
                                                missing_values=missing_values, axis_param=axis_param,
                                                field=field)

        return True if result else False
