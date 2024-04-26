
# Importing libraries
import numpy as np
import pandas as pd

# Importing functions and classes from packages
from helpers.auxiliar import cast_type_FixValue, find_closest_value, check_interval_condition
from helpers.invariant_aux import checkSpecialTypeMostFrequent, checkSpecialTypePrevious, checkSpecialTypeNext, \
    checkDerivedTypeColRowOutliers, checkSpecialTypeMedian, checkSpecialTypeInterpolation, checkSpecialTypeMean, \
    checkSpecialTypeClosest, checkIntervalMostFrequent, checkIntervalPrevious, checkIntervalNext, \
    checkFixValueMostFrequent, checkFixValuePrevious, checkFixValueNext, checkFixValueInterpolation, checkFixValueMean, \
    checkFixValueMedian, checkFixValueClosest
from helpers.transform_aux import getOutliers
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType, Belong




class Invariants:
    # FixValue - FixValue, FixValue - DerivedValue, FixValue - NumOp
    # Interval - FixValue, Interval - DerivedValue, Interval - NumOp
    # SpecialValue - FixValue, SpecialValue - DerivedValue, SpecialValue - NumOp

    def checkInv_FixValue_FixValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                   fixValueInput, fixValueOutput, belongOp_in: Belong = Belong.BELONG,
                                   belongOp_out: Belong = Belong.BELONG, dataTypeInput: DataType = None,
                                   dataTypeOutput: DataType = None, field: str = None) -> bool:
        """
        Check the invariant of the FixValue - FixValue relation (Mapping) is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        params:
            dataDictionary_in: dataframe with the input data
            dataDictionary_out: dataframe with the output data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            belongOp: condition to check the invariant
            dataTypeOutput: data type of the output value
            FixValueOutput: output value to check
            field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        if dataTypeInput is not None and dataTypeOutput is not None:  # If the data types are specified, the transformation is performed
            # Auxiliary function that changes the values of FixValueInput and FixValueOutput to the data type in DataTypeInput and DataTypeOutput respectively
            fixValueInput, fixValueOutput = cast_type_FixValue(dataTypeInput, fixValueInput, dataTypeOutput,
                                                               fixValueOutput)

        if field is None:
            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                # Iterar sobre las filas y columnas de dataDictionary_in
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        # Comprobar si el valor es igual a fixValueInput
                        if value == fixValueInput:
                            # Comprobar si el valor correspondiente en dataDictionary_out coincide con fixValueOutput
                            if dataDictionary_out.loc[row_index, column_name] != fixValueOutput:
                                return False
                        else: # Si el valor no es igual a fixValueInput
                            if dataDictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, column_name])):
                                return False
            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                # Iterar sobre las filas y columnas de dataDictionary_in
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        # Comprobar si el valor es igual a fixValueInput
                        if value == fixValueInput:
                            # Comprobar si el valor correspondiente en dataDictionary_out coincide con fixValueOutput
                            if dataDictionary_out.loc[row_index, column_name] == fixValueOutput:
                                return False
                        else: # Si el valor no es igual a fixValueInput
                            if dataDictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, column_name])):
                                return False
            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        # Comprobar si el valor es igual a fixValueInput
                        if value == fixValueInput:
                            return False
            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        # Comprobar si el valor es igual a fixValueInput
                        if value == fixValueInput:
                            return False

        elif field is not None:
            if field in dataDictionary_in.columns and field in dataDictionary_out.columns:
                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                    for row_index, value in dataDictionary_in[field].items():
                        # Comprobar si el valor es igual a fixValueInput
                        if value == fixValueInput:
                            # Comprobar si el valor correspondiente en dataDictionary_out coincide con fixValueOutput
                            if dataDictionary_out.loc[row_index, field] != fixValueOutput:
                                return False
                        else: # Si el valor no es igual a fixValueInput
                            if dataDictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, field])):
                                return False
                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                    for row_index, value in dataDictionary_in[field].items():
                        # Comprobar si el valor es igual a fixValueInput
                        if value == fixValueInput:
                            # Comprobar si el valor correspondiente en dataDictionary_out coincide con fixValueOutput
                            if dataDictionary_out.loc[row_index, field] == fixValueOutput:
                                return False
                        else: # Si el valor no es igual a fixValueInput
                            if dataDictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, field])):
                                return False
                elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
                    for row_index, value in dataDictionary_in[field].items():
                        # Comprobar si el valor es igual a fixValueInput
                        if value == fixValueInput:
                            return False
                elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
                    for row_index, value in dataDictionary_in[field].items():
                        # Comprobar si el valor es igual a fixValueInput
                        if value == fixValueInput:
                            return False
            elif field not in dataDictionary_in.columns or field not in dataDictionary_out.columns:
                raise ValueError("The field does not exist in the dataframe")

        return True

    def checkInv_FixValue_DerivedValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                       fixValueInput, derivedTypeOutput: DerivedType, belongOp_in: Belong = Belong.BELONG,
                                       belongOp_out: Belong = Belong.BELONG, dataTypeInput: DataType = None,
                                       axis_param: int = None, field: str = None) -> bool:
        # By default, if all values are equally frequent, it is replaced by the first value.
        # Check if it should only be done for rows and columns or also for the entire dataframe.
        """
        Check the invariant of the FixValue - DerivedValue relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        params:
            dataDictionary_in: dataframe with the input data
            dataDictionary_out: dataframe with the output data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            belongOp_in: if condition to check the invariant
            belongOp_out: then condition to check the invariant
            derivedTypeOutput: derived type of the output value
            axis_param: axis to check the invariant - 0: column, None: dataframe
            field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        if dataTypeInput is not None:  # If the data types are specified, the transformation is performed
            # Auxiliary function that changes the values of FixValueInput to the data type in DataTypeInput
            fixValueInput, fixValueOutput = cast_type_FixValue(dataTypeInput, fixValueInput, None,
                                                               None)

        result = True

        if derivedTypeOutput == DerivedType.MOSTFREQUENT:
            result = checkFixValueMostFrequent(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_in, belongOp_out, axis_param, field)
        elif derivedTypeOutput == DerivedType.PREVIOUS:
            result = checkFixValuePrevious(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_in, belongOp_out, axis_param, field)
        elif derivedTypeOutput == DerivedType.NEXT:
            result = checkFixValueNext(dataDictionary_in, dataDictionary_out, fixValueInput, belongOp_in, belongOp_out, axis_param, field)

        return True if result else False


    def checkInv_FixValue_NumOp(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                fixValueInput, numOpOutput: Operation, belongOp_in: Belong = Belong.BELONG,
                                belongOp_out: Belong = Belong.BELONG, dataTypeInput: DataType = None,
                                axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the FixValue - NumOp relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        params:
            dataDictionary_in: dataframe with the input data
            dataDictionary_out: dataframe with the output data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            belongOp_in: if condition to check the invariant
            belongOp_out: then condition to check the invariant
            numOpOutput: operation to check the invariant
            axis_param: axis to check the invariant
            field: field to check the invariant
        Returns:
            dataDictionary with the FixValueInput values replaced by the result of the operation numOpOutput
        """

        if dataTypeInput is not None:  # If the data types are specified, the transformation is performed
            # Auxiliary function that changes the values of FixValueInput to the data type in DataTypeInput
            fixValueInput, fixValueOutput = cast_type_FixValue(dataTypeInput, fixValueInput, None,
                                                               None)
        result = True

        if numOpOutput == Operation.INTERPOLATION:
            result = checkFixValueInterpolation(dataDictionary_in=dataDictionary_in,
                                               dataDictionary_out=dataDictionary_out, fixValueInput=fixValueInput,
                                               belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                               axis_param=axis_param, field=field)

        elif numOpOutput == Operation.MEAN:
            result = checkFixValueMean(dataDictionary_in=dataDictionary_in,
                                       dataDictionary_out=dataDictionary_out, fixValueInput=fixValueInput,
                                       belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                       axis_param=axis_param, field=field)
        elif numOpOutput == Operation.MEDIAN:
            result = checkFixValueMedian(dataDictionary_in=dataDictionary_in,
                                       dataDictionary_out=dataDictionary_out, fixValueInput=fixValueInput,
                                       belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                       axis_param=axis_param, field=field)
        elif numOpOutput == Operation.CLOSEST:
            result = checkFixValueClosest(dataDictionary_in=dataDictionary_in,
                                       dataDictionary_out=dataDictionary_out, fixValueInput=fixValueInput,
                                       belongOp_in=belongOp_in, belongOp_out=belongOp_out,
                                       axis_param=axis_param, field=field)

        return True if result else False



    def checkInv_Interval_FixValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                   leftMargin: float, rightMargin: float, closureType: Closure, fixValueOutput,
                                   belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                   dataTypeOutput: DataType = None, field: str = None) -> bool:
        """
        Check the invariant of the Interval - FixValue relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: dataframe with the data
            :param dataDictionary_out: dataframe with the data
            :param leftMargin: left margin of the interval
            :param rightMargin: right margin of the interval
            :param closureType: closure type of the interval
            :param dataTypeOutput: data type of the output value
            :param belongOp_in: if condition to check the invariant
            :param belongOp_out: then condition to check the invariant
            :param fixValueOutput: output value to check
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """

        if dataTypeOutput is not None:  # If it is specified, the transformation is performed
            vacio, fixValueOutput = cast_type_FixValue(None, None, dataTypeOutput, fixValueOutput)

        if field is None:
            if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if check_interval_condition(value, leftMargin, rightMargin, closureType):
                            if dataDictionary_out.loc[row_index, column_name] != fixValueOutput:
                                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                    return False
                                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                    return True
                        else: # Si el valor no es igual a fixValueInput
                            if dataDictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, column_name])):
                                return False
            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if check_interval_condition(value, leftMargin, rightMargin, closureType):
                            return False
            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
                for column_index, column_name in enumerate(dataDictionary_in.columns):
                    for row_index, value in dataDictionary_in[column_name].items():
                        if check_interval_condition(value, leftMargin, rightMargin, closureType):
                            return False

        elif field is not None:
            if field not in dataDictionary_in.columns or field not in dataDictionary_out.columns:
                raise ValueError("The field does not exist in the dataframe")
            if not np.issubdtype(dataDictionary_in[field].dtype, np.number):
                raise ValueError("The field is not numeric")

            if (belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG) or (belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG):
                for row_index, value in dataDictionary_in[field].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        if dataDictionary_out.loc[row_index, field] != fixValueOutput:
                            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                                return False
                            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                                return True
                    else: # Si el valor no es igual a fixValueInput
                        if dataDictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, field])):
                            return False
            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
                for row_index, value in dataDictionary_in[field].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False
            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
                for row_index, value in dataDictionary_in[field].items():
                    if check_interval_condition(value, leftMargin, rightMargin, closureType):
                        return False


        if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
            return True
        elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
            return False
        else:
            return True

    def checkInv_Interval_DerivedValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                       leftMargin: float, rightMargin: float,
                                       closureType: Closure, derivedTypeOutput: DerivedType,
                                       belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                       axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the Interval - DerivedValue relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: dataframe with the data
            :param dataDictionary_out: dataframe with the data
            :param leftMargin: left margin of the interval
            :param rightMargin: right margin of the interval
            :param closureType: closure type of the interval
            :param derivedTypeOutput: derived type of the output value
            :param belongOp_in: if condition to check the invariant
            :param belongOp_out: then condition to check the invariant
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        result = True

        if derivedTypeOutput == DerivedType.MOSTFREQUENT:
            result = checkIntervalMostFrequent(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                               closureType, belongOp_in, belongOp_out, axis_param, field)
        elif derivedTypeOutput == DerivedType.PREVIOUS:
            result = checkIntervalPrevious(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                           closureType, belongOp_in, belongOp_out, axis_param, field)
        elif derivedTypeOutput == DerivedType.NEXT:
            result = checkIntervalNext(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                       closureType, belongOp_in, belongOp_out, axis_param, field)

        return True if result else False

    def checkInv_Interval_NumOp(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                leftMargin: float, rightMargin: float, closureType: Closure, numOpOutput: Operation,
                                belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the FixValue - NumOp relation
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        params:
            :param dataDictionary_in: dataframe with the data
            :param dataDictionary_out: dataframe with the data
            :param leftMargin: left margin of the interval
            :param rightMargin: right margin of the interval
            :param closureType: closure type of the interval
            :param numOpOutput: operation to check the invariant
            :param belongOp_in: operation to check the invariant
            :param belongOp_out: operation to check the invariant
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """

        result = True

        if numOpOutput == Operation.INTERPOLATION:
            result = checkIntervalInterpolation(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                               closureType, belongOp_in, belongOp_out, axis_param, field)
        elif numOpOutput == Operation.MEAN:
            result = checkIntervalMean(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                           closureType, belongOp_in, belongOp_out, axis_param, field)
        elif numOpOutput == Operation.MEDIAN:
            result = checkIntervalMedian(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                       closureType, belongOp_in, belongOp_out, axis_param, field)
        elif numOpOutput == Operation.CLOSEST:
            result = checkIntervalClosest(dataDictionary_in, dataDictionary_out, leftMargin, rightMargin,
                                       closureType, belongOp_in, belongOp_out, axis_param, field)

        return True if result else False


    def checkInv_SpecialValue_FixValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                       specialTypeInput: SpecialType, fixValueOutput,
                                       belongOp_in: Belong = Belong.BELONG,
                                       belongOp_out: Belong = Belong.BELONG, dataTypeOutput: DataType = None,
                                       missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the SpecialValue - FixValue relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: input dataframe with the data
            :param dataDictionary_out: output dataframe with the data
            :param specialTypeInput: special type of the input value
            :param dataTypeOutput: data type of the output value
            :param fixValueOutput: output value to check
            :param belongOp_in: if condition to check the invariant
            :param belongOp_out: then condition to check the invariant
            :param missing_values: list of missing values
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        if dataTypeOutput is not None:  # If it is specified, the casting is performed
            vacio, fixValueOutput = cast_type_FixValue(None, None, dataTypeOutput, fixValueOutput)

        if field is None:
            if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                if specialTypeInput == SpecialType.MISSING:
                    for column_index, column_name in enumerate(dataDictionary_in.columns):
                        for row_index, value in dataDictionary_in[column_name].items():
                            if value in missing_values or pd.isnull(value):
                                if dataDictionary_out.loc[row_index, column_name] != fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[row_index, column_name] != value:
                                    return False
                elif specialTypeInput == SpecialType.INVALID:
                    for column_index, column_name in enumerate(dataDictionary_in.columns):
                        for row_index, value in dataDictionary_in[column_name].items():
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, column_name] != fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, column_name])):
                                    return False
                elif specialTypeInput == SpecialType.OUTLIER:
                    threshold = 1.5
                    if axis_param is None:
                        Q1 = dataDictionary_in.stack().quantile(0.25)
                        Q3 = dataDictionary_in.stack().quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the dataframe
                        numeric_values = dataDictionary_in.select_dtypes(include=[np.number])
                        for col in numeric_values.columns:
                            for idx in numeric_values.index:
                                value = numeric_values.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    if dataDictionary_out.loc[idx, col] != fixValueOutput:
                                        return False
                                else: # Si el valor no es igual a fixValueInput
                                    if dataDictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[idx, col])):
                                        return False
                    elif axis_param == 0:
                        # Iterate over each numeric column
                        for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                            # Calculate the Q1, Q3, and IQR for each column
                            Q1 = dataDictionary_in[col].quantile(0.25)
                            Q3 = dataDictionary_in[col].quantile(0.75)
                            IQR = Q3 - Q1
                            # Define the lower and upper bounds
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            # Identify the outliers in the column
                            for idx in dataDictionary_in.index:
                                value = dataDictionary_in.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    if dataDictionary_out.loc[idx, col] != fixValueOutput:
                                        return False
                                else: # Si el valor no es igual a fixValueInput
                                    if dataDictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[idx, col])):
                                        return False
                    elif axis_param == 1:
                        # Iterate over each row
                        for idx in dataDictionary_in.index:
                            # Calculate the Q1, Q3, and IQR for each row
                            Q1 = dataDictionary_in.loc[idx].quantile(0.25)
                            Q3 = dataDictionary_in.loc[idx].quantile(0.75)
                            IQR = Q3 - Q1
                            # Define the lower and upper bounds
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            # Identify the outliers in the row
                            for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                                value = dataDictionary_in.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    if dataDictionary_out.loc[idx, col] != fixValueOutput:
                                        return False
                                else: # Si el valor no es igual a fixValueInput
                                    if dataDictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[idx, col])):
                                        return False

            elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                if specialTypeInput == SpecialType.MISSING:
                    for column_index, column_name in enumerate(dataDictionary_in.columns):
                        for row_index, value in dataDictionary_in[column_name].items():
                            if value in missing_values or pd.isnull(value):
                                if dataDictionary_out.loc[row_index, column_name] == fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[row_index, column_name] != value:
                                    return False
                elif specialTypeInput == SpecialType.INVALID:
                    for column_index, column_name in enumerate(dataDictionary_in.columns):
                        for row_index, value in dataDictionary_in[column_name].items():
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, column_name] == fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[row_index, column_name] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, column_name])):
                                    return False
                elif specialTypeInput == SpecialType.OUTLIER:
                    threshold = 1.5
                    if axis_param is None:
                        Q1 = dataDictionary_in.stack().quantile(0.25)
                        Q3 = dataDictionary_in.stack().quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the dataframe
                        numeric_values = dataDictionary_in.select_dtypes(include=[np.number])
                        for col in numeric_values.columns:
                            for idx in numeric_values.index:
                                value = numeric_values.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    if dataDictionary_out.loc[idx, col] == fixValueOutput:
                                        return False
                                else: # Si el valor no es igual a fixValueInput
                                    if dataDictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[idx, col])):
                                        return False
                    elif axis_param == 0:
                        # Iterate over each numeric column
                        for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                            # Calculate the Q1, Q3, and IQR for each column
                            Q1 = dataDictionary_in[col].quantile(0.25)
                            Q3 = dataDictionary_in[col].quantile(0.75)
                            IQR = Q3 - Q1
                            # Define the lower and upper bounds
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            # Identify the outliers in the column
                            for idx in dataDictionary_in.index:
                                value = dataDictionary_in.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    if dataDictionary_out.loc[idx, col] == fixValueOutput:
                                        return False
                                else: # Si el valor no es igual a fixValueInput
                                    if dataDictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[idx, col])):
                                        return False
                    elif axis_param == 1:
                        # Iterate over each row
                        for idx in dataDictionary_in.index:
                            # Calculate the Q1, Q3, and IQR for each row
                            Q1 = dataDictionary_in.loc[idx].quantile(0.25)
                            Q3 = dataDictionary_in.loc[idx].quantile(0.75)
                            IQR = Q3 - Q1
                            # Define the lower and upper bounds
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            # Identify the outliers in the row
                            for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                                value = dataDictionary_in.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    if dataDictionary_out.loc[idx, col] == fixValueOutput:
                                        return False
                                else: # Si el valor no es igual a fixValueInput
                                    if dataDictionary_out.loc[idx, col] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[idx, col])):
                                        return False

            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
                if specialTypeInput == SpecialType.MISSING:
                    for column_index, column_name in enumerate(dataDictionary_in.columns):
                        for row_index, value in dataDictionary_in[column_name].items():
                            if value in missing_values or pd.isnull(value):
                                return False
                elif specialTypeInput == SpecialType.INVALID:
                    for column_index, column_name in enumerate(dataDictionary_in.columns):
                        for row_index, value in dataDictionary_in[column_name].items():
                            if value in missing_values:
                                return False
                elif specialTypeInput == SpecialType.OUTLIER:
                    threshold = 1.5
                    if axis_param is None:
                        Q1 = dataDictionary_in.stack().quantile(0.25)
                        Q3 = dataDictionary_in.stack().quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the dataframe
                        numeric_values = dataDictionary_in.select_dtypes(include=[np.number])
                        for col in numeric_values.columns:
                            for idx in numeric_values.index:
                                value = numeric_values.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    return False
                    elif axis_param == 0:
                        # Iterate over each numeric column
                        for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                            # Calculate the Q1, Q3, and IQR for each column
                            Q1 = dataDictionary_in[col].quantile(0.25)
                            Q3 = dataDictionary_in[col].quantile(0.75)
                            IQR = Q3 - Q1
                            # Define the lower and upper bounds
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            # Identify the outliers in the column
                            for idx in dataDictionary_in.index:
                                value = dataDictionary_in.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    return False
                    elif axis_param == 1:
                        # Iterate over each row
                        for idx in dataDictionary_in.index:
                            # Calculate the Q1, Q3, and IQR for each row
                            Q1 = dataDictionary_in.loc[idx].quantile(0.25)
                            Q3 = dataDictionary_in.loc[idx].quantile(0.75)
                            IQR = Q3 - Q1
                            # Define the lower and upper bounds
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            # Identify the outliers in the row
                            for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                                value = dataDictionary_in.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    return False

            elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
                if specialTypeInput == SpecialType.MISSING:
                    for column_index, column_name in enumerate(dataDictionary_in.columns):
                        for row_index, value in dataDictionary_in[column_name].items():
                            if value in missing_values or pd.isnull(value):
                                return False
                elif specialTypeInput == SpecialType.INVALID:
                    for column_index, column_name in enumerate(dataDictionary_in.columns):
                        for row_index, value in dataDictionary_in[column_name].items():
                            if value in missing_values:
                                return False
                elif specialTypeInput == SpecialType.OUTLIER:
                    threshold = 1.5
                    if axis_param is None:
                        Q1 = dataDictionary_in.stack().quantile(0.25)
                        Q3 = dataDictionary_in.stack().quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the dataframe
                        numeric_values = dataDictionary_in.select_dtypes(include=[np.number])
                        for col in numeric_values.columns:
                            for idx in numeric_values.index:
                                value = numeric_values.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    return False
                    elif axis_param == 0:
                        # Iterate over each numeric column
                        for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                            # Calculate the Q1, Q3, and IQR for each column
                            Q1 = dataDictionary_in[col].quantile(0.25)
                            Q3 = dataDictionary_in[col].quantile(0.75)
                            IQR = Q3 - Q1
                            # Define the lower and upper bounds
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            # Identify the outliers in the column
                            for idx in dataDictionary_in.index:
                                value = dataDictionary_in.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    return False
                    elif axis_param == 1:
                        # Iterate over each row
                        for idx in dataDictionary_in.index:
                            # Calculate the Q1, Q3, and IQR for each row
                            Q1 = dataDictionary_in.loc[idx].quantile(0.25)
                            Q3 = dataDictionary_in.loc[idx].quantile(0.75)
                            IQR = Q3 - Q1
                            # Define the lower and upper bounds
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            # Identify the outliers in the row
                            for col in dataDictionary_in.select_dtypes(include=[np.number]).columns:
                                value = dataDictionary_in.loc[idx, col]
                                is_outlier = (value < lower_bound) or (value > upper_bound)
                                if is_outlier:
                                    return False

        elif field is not None:
            if field in dataDictionary_in.columns and field in dataDictionary_out.columns:
                if belongOp_in == Belong.BELONG and belongOp_out == Belong.BELONG:
                    if specialTypeInput == SpecialType.MISSING:
                        for row_index, value in dataDictionary_in[field].items():
                            if value in missing_values or pd.isnull(value):
                                if dataDictionary_out.loc[row_index, field] != fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[row_index, field] != value:
                                    return False
                    elif specialTypeInput == SpecialType.INVALID:
                        for row_index, value in dataDictionary_in[field].items():
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, field] != fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, field])):
                                    return False
                    elif specialTypeInput == SpecialType.OUTLIER:
                        threshold = 1.5
                        # Calculate the Q1, Q3, and IQR for each column
                        Q1 = dataDictionary_in[field].quantile(0.25)
                        Q3 = dataDictionary_in[field].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the column
                        for idx in dataDictionary_in.index:
                            value = dataDictionary_in.loc[idx, field]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if dataDictionary_out.loc[idx, field] != fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[idx, field] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[idx, field])):
                                    return False

                elif belongOp_in == Belong.BELONG and belongOp_out == Belong.NOTBELONG:
                    if specialTypeInput == SpecialType.MISSING:
                        for row_index, value in dataDictionary_in[field].items():
                            if value in missing_values or pd.isnull(value):
                                if dataDictionary_out.loc[row_index, field] == fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[row_index, field] != value:
                                    return False
                    elif specialTypeInput == SpecialType.INVALID:
                        for row_index, value in dataDictionary_in[field].items():
                            if value in missing_values:
                                if dataDictionary_out.loc[row_index, field] == fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[row_index, field] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[row_index, field])):
                                    return False
                    elif specialTypeInput == SpecialType.OUTLIER:
                        threshold = 1.5
                        # Calculate the Q1, Q3, and IQR for each column
                        Q1 = dataDictionary_in[field].quantile(0.25)
                        Q3 = dataDictionary_in[field].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the column
                        for idx in dataDictionary_in.index:
                            value = dataDictionary_in.loc[idx, field]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                if dataDictionary_out.loc[idx, field] == fixValueOutput:
                                    return False
                            else: # Si el valor no es igual a fixValueInput
                                if dataDictionary_out.loc[idx, field] != value and not(pd.isnull(value) and pd.isnull(dataDictionary_out.loc[idx, field])):
                                    return False

                elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.BELONG:
                    if specialTypeInput == SpecialType.MISSING:
                        for row_index, value in dataDictionary_in[field].items():
                            if value in missing_values or pd.isnull(value):
                                return False
                    elif specialTypeInput == SpecialType.INVALID:
                        for row_index, value in dataDictionary_in[field].items():
                            if value in missing_values:
                                return False
                    elif specialTypeInput == SpecialType.OUTLIER:
                        threshold = 1.5
                        # Calculate the Q1, Q3, and IQR for each column
                        Q1 = dataDictionary_in[field].quantile(0.25)
                        Q3 = dataDictionary_in[field].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the column
                        for idx in dataDictionary_in.index:
                            value = dataDictionary_in.loc[idx, field]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                return False

                elif belongOp_in == Belong.NOTBELONG and belongOp_out == Belong.NOTBELONG:
                    if specialTypeInput == SpecialType.MISSING:
                        for row_index, value in dataDictionary_in[field].items():
                            if value in missing_values or pd.isnull(value):
                                return False
                    elif specialTypeInput == SpecialType.INVALID:
                        for row_index, value in dataDictionary_in[field].items():
                            if value in missing_values:
                                return False
                    elif specialTypeInput == SpecialType.OUTLIER:
                        threshold = 1.5
                        # Calculate the Q1, Q3, and IQR for each column
                        Q1 = dataDictionary_in[field].quantile(0.25)
                        Q3 = dataDictionary_in[field].quantile(0.75)
                        IQR = Q3 - Q1
                        # Define the lower and upper bounds
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        # Identify the outliers in the column
                        for idx in dataDictionary_in.index:
                            value = dataDictionary_in.loc[idx, field]
                            is_outlier = (value < lower_bound) or (value > upper_bound)
                            if is_outlier:
                                return False

            elif field not in dataDictionary_in.columns or field not in dataDictionary_out.columns:
                raise ValueError("The field does not exist in the dataframe")

        return True

    def checkInv_SpecialValue_DerivedValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                           specialTypeInput: SpecialType, derivedTypeOutput: DerivedType,
                                           belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                           missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the SpecialValue - DerivedValue relation
        params:
            :param dataDictionary_in: dataframe with the data
            :param dataDictionary_out: dataframe with the data
            :param specialTypeInput: special type of the input value
            :param derivedTypeOutput: derived type of the output value
            :param belongOp_in: if condition to check the invariant
            :param belongOp_out: then condition to check the invariant
            :param missing_values: list of missing values
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """

        dataDictionary_outliers_mask = None
        result = True

        if specialTypeInput == SpecialType.OUTLIER:
            dataDictionary_outliers_mask = getOutliers(dataDictionary_in, field, axis_param)

            if axis_param is None:
                missing_values = dataDictionary_in.where(dataDictionary_outliers_mask == 1).stack().tolist()

        if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                result = checkSpecialTypeMostFrequent(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                                      belongOp_in, belongOp_out, missing_values, axis_param, field)
            elif derivedTypeOutput == DerivedType.PREVIOUS:
                result = checkSpecialTypePrevious(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                                  belongOp_in, belongOp_out, missing_values, axis_param, field)
            elif derivedTypeOutput == DerivedType.NEXT:
                result = checkSpecialTypeNext(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                              belongOp_in, belongOp_out, missing_values, axis_param, field)

        elif specialTypeInput == SpecialType.OUTLIER:
            # IMPORTANT: The function getOutliers() does the same as apply_derivedTypeOutliers() but at the dataframe level.
            # If the outliers are applied at the dataframe level, previous and next cannot be applied.

            if axis_param is None:
                missing_values = dataDictionary_in.where(dataDictionary_outliers_mask == 1).stack().tolist()
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    result = checkSpecialTypeMostFrequent(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                                          belongOp_in, belongOp_out, missing_values, axis_param, field)
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    result = checkSpecialTypePrevious(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                                      belongOp_in, belongOp_out, missing_values, axis_param, field)
                elif derivedTypeOutput == DerivedType.NEXT:
                    result = checkSpecialTypeNext(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                                  belongOp_in, belongOp_out, missing_values, axis_param, field)

            elif axis_param == 0 or axis_param == 1:
                result = checkDerivedTypeColRowOutliers(derivedTypeOutput, dataDictionary_in,
                                                        dataDictionary_out,
                                                        dataDictionary_outliers_mask, belongOp_in, belongOp_out,
                                                        axis_param, field)
        elif field is not None:
            if field not in dataDictionary_in.columns:
                raise ValueError("The field does not exist in the dataframe")
            elif field in dataDictionary_in.columns:
                if np.issubdtype(dataDictionary_in[field].dtype, np.number):
                    if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
                        if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                            result = checkSpecialTypeMostFrequent(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                                                  belongOp_in, belongOp_out, missing_values, axis_param,
                                                                  field)
                        elif derivedTypeOutput == DerivedType.PREVIOUS:
                            result = checkSpecialTypePrevious(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                                              belongOp_in, belongOp_out, missing_values, axis_param,
                                                              field)
                        elif derivedTypeOutput == DerivedType.NEXT:
                            result = checkSpecialTypeNext(dataDictionary_in, dataDictionary_out, specialTypeInput,
                                                          belongOp_in, belongOp_out, missing_values, axis_param,
                                                          field)
                    elif specialTypeInput == SpecialType.OUTLIER:
                        result = checkDerivedTypeColRowOutliers(derivedTypeOutput, dataDictionary_in,
                                                                dataDictionary_out, dataDictionary_outliers_mask,
                                                                belongOp_in, belongOp_out, axis_param,
                                                                field)
                else:
                    raise ValueError("The field is not numeric")

        return True if result else False

    def checkInv_SpecialValue_NumOp(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                    specialTypeInput: SpecialType, numOpOutput: Operation,
                                    belongOp_in: Belong = Belong.BELONG, belongOp_out: Belong = Belong.BELONG,
                                    missing_values: list = None, axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the SpecialValue - NumOp relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: dataframe with the data
            :param dataDictionary_out: dataframe with the data
            :param specialTypeInput: special type of the input value
            :param numOpOutput: operation to check the invariant
            :param belongOp_in: if condition to check the invariant
            :param belongOp_out: then condition to check the invariant
            :param missing_values: list of missing values
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """

        dataDictionary_outliers_mask = None
        result = True

        if specialTypeInput == SpecialType.OUTLIER:
            dataDictionary_outliers_mask = getOutliers(dataDictionary_in, field, axis_param)

        if numOpOutput == Operation.INTERPOLATION:
            result = checkSpecialTypeInterpolation(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        specialTypeInput=specialTypeInput, belongOp_in=belongOp_in,
                                                        belongOp_out=belongOp_out,
                                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                        missing_values=missing_values, axis_param=axis_param,
                                                        field=field)
        elif numOpOutput == Operation.MEAN:
            result = checkSpecialTypeMean(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        specialTypeInput=specialTypeInput, belongOp_in=belongOp_in,
                                                        belongOp_out=belongOp_out,
                                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                        missing_values=missing_values, axis_param=axis_param,
                                                        field=field)
        elif numOpOutput == Operation.MEDIAN:
            result = checkSpecialTypeMedian(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        specialTypeInput=specialTypeInput, belongOp_in=belongOp_in,
                                                        belongOp_out=belongOp_out,
                                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                        missing_values=missing_values, axis_param=axis_param,
                                                        field=field)
        elif numOpOutput == Operation.CLOSEST:
            result = checkSpecialTypeClosest(dataDictionary_in=dataDictionary_in, dataDictionary_out=dataDictionary_out,
                                                        specialTypeInput=specialTypeInput, belongOp_in=belongOp_in,
                                                        belongOp_out=belongOp_out,
                                                        dataDictionary_outliers_mask=dataDictionary_outliers_mask,
                                                        missing_values=missing_values, axis_param=axis_param,
                                                        field=field)

        return True if result else False
