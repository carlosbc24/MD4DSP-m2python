import numpy as np
import pandas as pd

from helpers.auxiliar import cast_type_FixValue, find_closest_value, getOutliers, apply_derivedTypeColRowOutliers, \
    apply_derivedType, specialTypeInterpolation, specialTypeMean, specialTypeMedian, specialTypeClosest

# Importing functions and classes from packages
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType, Belong


class Invariants:
    # FixValue - FixValue, FixValue - DerivedValue, FixValue - NumOp
    # Interval - FixValue, Interval - DerivedValue, Interval - NumOp
    # SpecialValue - FixValue, SpecialValue - DerivedValue, SpecialValue - NumOp

    def checkInv_FixValue_FixValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                   fixValueInput, fixValueOutput, belongOp: Belong = Belong.BELONG, dataTypeInput: DataType = None,
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
            # Iterar sobre las filas y columnas de dataDictionary_in
            for column_index, column_name in enumerate(dataDictionary_in.columns):
                for row_index, value in dataDictionary_in[column_name].items():
                    # Comprobar si el valor es igual a fixValueInput
                    if value == fixValueInput:
                        # Comprobar si el valor correspondiente en dataDictionary_out coincide con fixValueOutput
                        if dataDictionary_out.loc[row_index, column_name] != fixValueOutput:
                            return False
        elif field is not None:
            if field in dataDictionary_in.columns and field in dataDictionary_out.columns:
                for row_index, value in dataDictionary_in[field].items():
                    # Comprobar si el valor es igual a fixValueInput
                    if value == fixValueInput:
                        # Comprobar si el valor correspondiente en dataDictionary_out coincide con fixValueOutput
                        if dataDictionary_out.loc[row_index, field] != fixValueOutput:
                            return False
            elif field not in dataDictionary_in.columns or dataDictionary_out.columns:
                raise ValueError("The field does not exist in the dataframe")

        return True

    def checkInv_FixValue_DerivedValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                       fixValueInput, derivedTypeOutput: DerivedType, belongOp: Belong = Belong.BELONG,
                                       dataTypeInput: DataType = None, axis_param: int = None,
                                       field: str = None) -> bool:
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
            belongOp: operation to check the invariant
            derivedTypeOutput: derived type of the output value
            axis_param: axis to check the invariant - 0: column, None: dataframe
            field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        return True

    def checkInv_FixValue_NumOp(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                fixValueInput, numOpOutput: Operation, belongOp: Belong = Belong.BELONG,
                                dataTypeInput: DataType = None, axis_param: int = None,
                                field: str = None) -> bool:
        """
        Check the invariant of the FixValue - NumOp relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        params:
            dataDictionary_in: dataframe with the input data
            dataDictionary_out: dataframe with the output data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            belongOp: operation to check the invariant
            numOpOutput: operation to check the invariant
            axis_param: axis to check the invariant
            field: field to check the invariant
        Returns:
            dataDictionary with the FixValueInput values replaced by the result of the operation numOpOutput
        """
        return True

    def checkInv_Interval_FixValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                   leftMargin: float, rightMargin: float, closureType: Closure, fixValueOutput,
                                   belongOp: Belong = Belong.BELONG, dataTypeOutput: DataType = None,
                                   field: str = None) -> bool:
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
            :param belongOp: operation to check the invariant
            :param fixValueOutput: output value to check
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        return True

    def checkInv_Interval_DerivedValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                       leftMargin: float, rightMargin: float,
                                       closureType: Closure, derivedTypeOutput: DerivedType,
                                       belongOp: Belong = Belong.BELONG,
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
            :param belongOp: operation to check the invariant
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        return True

    def checkInv_Interval_NumOp(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                leftMargin: float, rightMargin: float, closureType: Closure, numOpOutput: Operation,
                                belongOp: Belong = Belong.BELONG, axis_param: int = None, field: str = None) -> bool:
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
            :param belongOp: operation to check the invariant
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        return True

    def checkInv_SpecialValue_FixValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                       specialTypeInput: SpecialType, fixValueOutput, belongOp: Belong = Belong.BELONG,
                                       dataTypeOutput: DataType = None, missing_values: list = None,
                                       axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the SpecialValue - FixValue relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: dataframe with the data
            :param dataDictionary_out: dataframe with the data
            :param specialTypeInput: special type of the input value
            :param dataTypeOutput: data type of the output value
            :param fixValueOutput: output value to check
            :param belongOp: operation to check the invariant
            :param missing_values: list of missing values
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        return True

    def checkInv_SpecialValue_DerivedValue(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                           specialTypeInput: SpecialType, derivedTypeOutput: DerivedType,
                                           belongOp: Belong = Belong.BELONG, missing_values: list = None,
                                           axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the SpecialValue - DerivedValue relation
        params:
            :param dataDictionary_in: dataframe with the data
            :param dataDictionary_out: dataframe with the data
            :param specialTypeInput: special type of the input value
            :param derivedTypeOutput: derived type of the output value
            :param belongOp: operation to check the invariant
            :param missing_values: list of missing values
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        return True

    def checkInv_SpecialValue_NumOp(self, dataDictionary_in: pd.DataFrame, dataDictionary_out: pd.DataFrame,
                                    specialTypeInput: SpecialType, numOpOutput: Operation,
                                    belongOp: Belong = Belong.BELONG, missing_values: list = None,
                                    axis_param: int = None, field: str = None) -> bool:
        """
        Check the invariant of the SpecialValue - NumOp relation is satisfied in the dataDicionary_out
        respect to the dataDictionary_in
        params:
            :param dataDictionary_in: dataframe with the data
            :param dataDictionary_out: dataframe with the data
            :param specialTypeInput: special type of the input value
            :param numOpOutput: operation to check the invariant
            :param belongOp: operation to check the invariant
            :param missing_values: list of missing values
            :param axis_param: axis to check the invariant
            :param field: field to check the invariant

        returns:
            True if the invariant is satisfied, False otherwise
        """
        return True
