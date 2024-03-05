# Importing libraries
import numpy as np
import pandas as pd

from helpers.auxiliar import cast_type_FixValue, find_closest_value, getOutliers, apply_derivedTypeColRowOutliers, \
    apply_derivedType, specialTypeInterpolation, specialTypeMean, specialTypeMedian, specialTypeClosest
# Importing functions and classes from packages
from helpers.enumerations import Closure, DataType, DerivedType, Operation, SpecialType


class ContractsInvariants:
    # FixValue - FixValue, FixValue - DerivedValue, FixValue - NumOp
    # Interval - FixValue, Interval - DerivedValue, Interval - NumOp
    # SpecialValue - FixValue, SpecialValue - DerivedValue, SpecialValue - NumOp

    def checkInv_FixValue_FixValue(self, dataDictionary: pd.DataFrame, fixValueInput, fixValueOutput,
                                   dataTypeInput: DataType = None,
                                   dataTypeOutput: DataType = None, field: str = None) -> pd.DataFrame:
        """
        Check the invariant of the FixValue - FixValue relation
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            dataTypeOutput: data type of the output value
            FixValueOutput: output value to check
            field: field to check the invariant
        Returns:
            dataDictionary with the FixValueInput and FixValueOutput values changed to the type dataTypeInput and dataTypeOutput respectively
        """
        if dataTypeInput is not None and dataTypeOutput is not None:  # Si se especifican los tipos de dato, se realiza la transformación
            # Función auxiliar que cambia los valores de FixValueInput y FixValueOutput al tipo de dato en DataTypeInput y DataTypeOutput respectivamente
            fixValueInput, fixValueOutput = cast_type_FixValue(dataTypeInput, fixValueInput, dataTypeOutput,
                                                               fixValueOutput)

        if field is None:
            dataDictionary = dataDictionary.replace(fixValueInput, fixValueOutput)
        elif field is not None:
            if field in dataDictionary.columns:
                dataDictionary.loc[:, field] = dataDictionary.loc[:, field].replace(fixValueInput, fixValueOutput)
                # dataDictionary[field] = dataDictionary[field].replace(fixValueInput, fixValueOutput)
            elif field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

        return dataDictionary

    def checkInv_FixValue_DerivedValue(self, dataDictionary: pd.DataFrame, fixValueInput,
                                       derivedTypeOutput: DerivedType,
                                       dataTypeInput: DataType = None, axis_param: int = None,
                                       field: str = None) -> pd.DataFrame:
        # Por defecto, si todos los valores son igual de frecuentes, se sustituye por el primer valor.
        # Comprobar si solo se debe hacer para filas y columnas o también para el dataframe completo.
        """
        Check the invariant of the FixValue - DerivedValue relation
        Sustituye el valor proporcionado por el usuario por el valor derivado en el eje que se especifique por parámetros
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            derivedTypeOutput: derived type of the output value
            axis_param: axis to check the invariant
            field: field to check the invariant

            return: dataDictionary with the FixValueInput values replaced by the value derived from the operation derivedTypeOutput
        """
        if dataTypeInput is not None:  # Si se especifica el tipo de dato, se realiza la transformación
            fixValueInput, valorNulo = cast_type_FixValue(dataTypeInput, fixValueInput, None, None)
        # Función auxiliar que cambia el valor de FixValueInput al tipo de dato en DataTypeInput

        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                if axis_param == 1:  # Aplica la función lambda a nivel de fila
                    dataDictionary_copy = dataDictionary_copy.apply(lambda fila: fila.apply(
                        lambda value: dataDictionary_copy.loc[
                            fila.name].value_counts().idxmax() if value == fixValueInput else value), axis=axis_param)
                elif axis_param == 0:  # Aplica la función lambda a nivel de columna
                    dataDictionary_copy = dataDictionary_copy.apply(lambda columna: columna.apply(
                        lambda value: dataDictionary_copy[
                            columna.name].value_counts().idxmax() if value == fixValueInput else value),
                                                                    axis=axis_param)
                else:  # Aplica la función lambda a nivel de dataframe
                    # En caso de empate de valor con más apariciones en el dataset, se toma el primer valor
                    valor_mas_frecuente = dataDictionary_copy.stack().value_counts().idxmax()
                    # Reemplaza 'fixValueInput' con el valor más frecuente en el DataFrame completo usando lambda
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.replace(fixValueInput, valor_mas_frecuente))
            # Si el valor es el primero, se asigna a np.nan
            elif derivedTypeOutput == DerivedType.PREVIOUS:
                # Aplica la función lambda a nivel de columna (axis=0) o a nivel de fila (axis=1)
                # Lambda que sustitutuye cualquier valor igual a FixValueInput del dataframe por el valor de la fila anterior en la misma columna
                if axis_param is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda x: x.where((x != fixValueInput) | x.shift(1).isna(),
                                          other=x.shift(1)), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the PREVIOUS operation")
            # Si el valor es el último, se asigna a np.nan
            elif derivedTypeOutput == DerivedType.NEXT:
                # Aplica la función lambda a nivel de columna (axis=0) o a nivel de fila (axis=1)
                if axis_param is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda x: x.where((x != fixValueInput) | x.shift(-1).isna(),
                                          other=x.shift(-1)), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the NEXT operation")
        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")
            elif field in dataDictionary.columns:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(lambda value: dataDictionary_copy[
                        field].value_counts().idxmax() if value == fixValueInput else value)
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    dataDictionary_copy[field] = dataDictionary_copy[field].where(
                        (dataDictionary_copy[field] != fixValueInput) | dataDictionary_copy[field].shift(1).isna(),
                        other=dataDictionary_copy[field].shift(1))
                elif derivedTypeOutput == DerivedType.NEXT:
                    dataDictionary_copy[field] = dataDictionary_copy[field].where(
                        (dataDictionary_copy[field] != fixValueInput) | dataDictionary_copy[field].shift(-1).isna(),
                        other=dataDictionary_copy[field].shift(-1))

        return dataDictionary_copy

    def checkInv_FixValue_NumOp(self, dataDictionary: pd.DataFrame, fixValueInput, numOpOutput: Operation,
                                dataTypeInput: DataType = None, axis_param: int = None,
                                field: str = None) -> pd.DataFrame:
        """
        Check the invariant of the FixValue - NumOp relation
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        params:
            dataDictionary: dataframe with the data
            dataTypeInput: data type of the input value
            FixValueInput: input value to check
            numOpOutput: operation to check the invariant
            axis_param: axis to check the invariant
            field: field to check the invariant
        Returns:
            dataDictionary with the FixValueInput values replaced by the result of the operation numOpOutput
        """
        if dataTypeInput is not None:  # Si se especifica el tipo de dato, se realiza la transformación
            fixValueInput, valorNulo = cast_type_FixValue(dataTypeInput, fixValueInput, None, None)

        # Función auxiliar que cambia el valor de FixValueInput al tipo de dato en DataTypeInput
        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if numOpOutput == Operation.INTERPOLATION:
                # Aplicamos la interpolación lineal en el DataFrame
                if axis_param == 0:
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                            dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                                lambda x: np.nan if x == fixValueInput else x).interpolate(method='linear',
                                                                                           limit_direction='both')
                elif axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda row: row.apply(lambda x: np.nan if x == fixValueInput else x).interpolate(
                            method='linear', limit_direction='both'), axis=axis_param)
                elif axis_param is None:
                    raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

            elif numOpOutput == Operation.MEAN:
                if axis_param == None:
                    # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    # Calcular la media de estas columnas numéricas
                    mean_value = only_numbers_df.mean().mean()
                    # Reemplaza 'fixValueInput' con la media del DataFrame completo usando lambda
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.replace(fixValueInput, mean_value))
                elif axis_param == 0:
                    means = dataDictionary_copy.apply(
                        lambda col: col[col.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                            col.dtype, np.number) else None)

                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                            dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                                lambda x: x if x != fixValueInput else means[col])
                elif axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.T

                    means = dataDictionary_copy.apply(
                        lambda row: row[row.apply(lambda x: np.issubdtype(type(x), np.number))].mean() if np.issubdtype(
                            row.dtype, np.number) else None)

                    for row in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[row].dtype, np.number):
                            dataDictionary_copy[row] = dataDictionary_copy[row].apply(
                                lambda x: x if x != fixValueInput else means[row])

                    dataDictionary_copy = dataDictionary_copy.T

            elif numOpOutput == Operation.MEDIAN:
                if axis_param == None:
                    # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    # Calcular la media de estas columnas numéricas
                    median_value = only_numbers_df.median().median()
                    # Reemplaza 'fixValueInput' con la mediana del DataFrame completo usando lambda
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.replace(fixValueInput, median_value))
                elif axis_param == 0 or axis_param == 1:
                    # dataDictionary_copy = dataDictionary_copy.apply(lambda x: x.where(x != fixValueInput, other=x.mean()), axis=axis_param)
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda x: x.apply(
                            lambda y: y if not np.issubdtype(type(y), np.number) or y != fixValueInput
                            else x[x.apply(lambda z: np.issubdtype(type(z), np.number))].median()), axis=axis_param)

            elif numOpOutput == Operation.CLOSEST:
                if axis_param is None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(lambda x: find_closest_value(dataDictionary_copy.stack(),
                                                                           fixValueInput) if x == fixValueInput else x))
                elif axis_param == 0 or axis_param == 1:
                    # Reemplazar 'fixValueInput' por el valor numérico más cercano a lo largo de las columnas
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: find_closest_value(col, fixValueInput) if x == fixValueInput else x),
                        axis=axis_param)

            else:
                raise ValueError("No valid operator")
        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                    if numOpOutput == Operation.INTERPOLATION:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if x != fixValueInput else np.nan).interpolate(method='linear',
                                                                                       limit_direction='both')
                    elif numOpOutput == Operation.MEAN:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if x != fixValueInput else dataDictionary_copy[field].mean())
                    elif numOpOutput == Operation.MEDIAN:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if x != fixValueInput else dataDictionary_copy[field].median())
                    elif numOpOutput == Operation.CLOSEST:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if x != fixValueInput else find_closest_value(dataDictionary_copy[field],
                                                                                      fixValueInput))
                else:
                    raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def checkInv_Interval_FixValue(self, dataDictionary: pd.DataFrame, leftMargin: float, rightMargin: float,
                                   closureType: Closure, fixValueOutput, dataTypeOutput: DataType = None,
                                   field: str = None) -> pd.DataFrame:
        """
        Check the invariant of the Interval - FixValue relation
        :param dataDictionary: dataframe with the data
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param closureType: closure type of the interval
        :param dataTypeOutput: data type of the output value
        :param fixValueOutput: output value to check
        :param field: field to check the invariant
        :return: dataDictionary with the values of the interval changed to the value fixValueOutput
        """
        if dataTypeOutput is not None:  # Si se especifica el tipo de dato, se realiza la transformación
            vacio, fixValueOutput = cast_type_FixValue(None, None, dataTypeOutput, fixValueOutput)

        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            # Aplicar el cambio en los valores dentro del intervalo
            if closureType == Closure.openOpen:
                dataDictionary_copy = dataDictionary.apply(lambda func: func.apply(lambda x: fixValueOutput if (
                                np.issubdtype(type(x), np.number) and leftMargin < x < rightMargin) else x))
            elif closureType == Closure.openClosed:
                dataDictionary_copy = dataDictionary.apply(lambda func: func.apply(lambda x: fixValueOutput if
                                np.issubdtype(type(x), np.number) and (leftMargin < x) and (x <= rightMargin) else x))
            elif closureType == Closure.closedOpen:
                dataDictionary_copy = dataDictionary.apply(lambda func: func.apply(lambda x: fixValueOutput if
                                np.issubdtype(type(x), np.number) and (leftMargin <= x) and (x < rightMargin) else x))
            elif closureType == Closure.closedClosed:
                dataDictionary_copy = dataDictionary.apply(lambda func: func.apply(lambda x: fixValueOutput if
                                np.issubdtype(type(x), np.number) and (leftMargin <= x) and (x <= rightMargin) else x))

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if np.issubdtype(dataDictionary[field].dtype, np.number):
                    if closureType == Closure.openOpen:
                        dataDictionary_copy[field] = dataDictionary[field].apply(
                            lambda x: fixValueOutput if (leftMargin < x) and (x < rightMargin) else x)
                    elif closureType == Closure.openClosed:
                        dataDictionary_copy[field] = dataDictionary[field].apply(
                            lambda x: fixValueOutput if (leftMargin < x) and (x <= rightMargin) else x)
                    elif closureType == Closure.closedOpen:
                        dataDictionary_copy[field] = dataDictionary[field].apply(
                            lambda x: fixValueOutput if (leftMargin <= x) and (x < rightMargin) else x)
                    elif closureType == Closure.closedClosed:
                        dataDictionary_copy[field] = dataDictionary[field].apply(
                            lambda x: fixValueOutput if (leftMargin <= x) and (x <= rightMargin) else x)
                else:
                    raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def checkInv_Interval_DerivedValue(self, dataDictionary: pd.DataFrame, leftMargin: float, rightMargin: float,
                                       closureType: Closure, derivedTypeOutput: DerivedType,
                                       axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Check the invariant of the Interval - DerivedValue relation
        :param dataDictionary: dataframe with the data
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param closureType: closure type of the interval
        :param derivedTypeOutput: derived type of the output value
        :param axis_param: axis to check the invariant
        :param field: field to check the invariant

        :return: dataDictionary with the values of the interval changed to the
            value derived from the operation derivedTypeOutput
        """
        dataDictionary_copy = dataDictionary.copy()

        # Definir la condición del intervalo basada en el tipo de cierre
        def get_condition(x):
            if closureType == Closure.openOpen:
                return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x < rightMargin)) else False
            elif closureType == Closure.openClosed:
                return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x <= rightMargin)) else False
            elif closureType == Closure.closedOpen:
                return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x < rightMargin)) else False
            elif closureType == Closure.closedClosed:
                return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x <= rightMargin)) else False

        if field is None:
            if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                if axis_param == 1:  # Aplica la función lambda a nivel de fila
                    dataDictionary_copy = dataDictionary_copy.apply(lambda fila: fila.apply(
                        lambda value: dataDictionary_copy.loc[
                            fila.name].value_counts().idxmax() if get_condition(value) else value), axis=axis_param)
                elif axis_param == 0:  # Aplica la función lambda a nivel de columna
                    dataDictionary_copy = dataDictionary_copy.apply(lambda columna: columna.apply(
                        lambda value: dataDictionary_copy[
                            columna.name].value_counts().idxmax() if get_condition(value) else value), axis=axis_param)
                else:  # Aplica la función lambda a nivel de dataframe
                    # En caso de empate de valor con más apariciones en el dataset, se toma el primer valor
                    valor_mas_frecuente = dataDictionary_copy.stack().value_counts().idxmax()
                    # Reemplaza los valores dentro del intervalo con el valor más frecuente en el DataFrame completo usando lambda
                    dataDictionary_copy = dataDictionary_copy.apply(lambda columna: columna.apply(
                        lambda value: valor_mas_frecuente if get_condition(value) else value))
            # No asigna nada a np.nan
            elif derivedTypeOutput == DerivedType.PREVIOUS:
                # Aplica la función lambda a nivel de columna (axis=0) o a nivel de fila (axis=1)
                # Lambda que sustitutuye cualquier valor igual a FixValueInput del dataframe por el valor de la fila anterior en la misma columna
                if axis_param is not None:
                    # Define una función lambda para reemplazar los valores dentro del intervalo por el valor de la
                    # posición anterior
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i - 1] if get_condition(value) and i > 0 else value
                                  for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the PREVIOUS operation")
            # No asigna nada a np.nan
            elif derivedTypeOutput == DerivedType.NEXT:
                # Aplica la función lambda a nivel de columna (axis=0) o a nivel de fila (axis=1)
                if axis_param is not None:
                    # Define la función lambda para reemplazar los valores dentro del intervalo por el valor de la siguiente posición
                    dataDictionary_copy = dataDictionary_copy.apply(lambda row_or_col: pd.Series([np.nan if pd.isnull(
                        value) else row_or_col.iloc[i + 1] if get_condition(value) and i < len(row_or_col) - 1 else value
                               for i, value in enumerate(row_or_col)], index=row_or_col.index), axis=axis_param)
                else:
                    raise ValueError("The axis cannot be None when applying the NEXT operation")

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if derivedTypeOutput == DerivedType.MOSTFREQUENT:
                    dataDictionary_copy[field] = dataDictionary_copy[field].apply(lambda value:
                                                dataDictionary_copy[field].value_counts().idxmax() if get_condition(value)
                                                                                                    else value)
                elif derivedTypeOutput == DerivedType.PREVIOUS:
                    dataDictionary_copy[field] = pd.Series(
                        [np.nan if pd.isnull(value) else dataDictionary_copy[field].iloc[i - 1]
                        if get_condition(value) and i > 0 else value for i, value in
                         enumerate(dataDictionary_copy[field])],
                        index=dataDictionary_copy[field].index)
                elif derivedTypeOutput == DerivedType.NEXT:
                    dataDictionary_copy[field] = pd.Series(
                        [np.nan if pd.isnull(value) else dataDictionary_copy[field].iloc[i + 1]
                        if get_condition(value) and i < len(dataDictionary_copy[field]) - 1 else value for i, value in
                         enumerate(dataDictionary_copy[field])], index=dataDictionary_copy[field].index)

        return dataDictionary_copy

    def checkInv_Interval_NumOp(self, dataDictionary: pd.DataFrame, leftMargin: float, rightMargin: float,
                                closureType: Closure, numOpOutput: Operation, axis_param: int = None,
                                field: str = None) -> pd.DataFrame:
        """
        Check the invariant of the FixValue - NumOp relation
        If the value of 'axis_param' is None, the operation mean or median is applied to the entire dataframe
        :param dataDictionary: dataframe with the data
        :param leftMargin: left margin of the interval
        :param rightMargin: right margin of the interval
        :param numOpOutput: operation to check the invariant
        :param axis_param: axis to check the invariant
        :param field: field to check the invariant
        :return: dataDictionary with the values of the interval changed to the result of the operation numOpOutput
        """

        def get_condition(x):
            if closureType == Closure.openOpen:
                return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x < rightMargin)) else False
            elif closureType == Closure.openClosed:
                return True if np.issubdtype(type(x), np.number) and ((x > leftMargin) & (x <= rightMargin)) else False
            elif closureType == Closure.closedOpen:
                return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x < rightMargin)) else False
            elif closureType == Closure.closedClosed:
                return True if np.issubdtype(type(x), np.number) and ((x >= leftMargin) & (x <= rightMargin)) else False

        # Función auxiliar que cambia el valor de FixValueInput al tipo de dato en DataTypeInput
        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if numOpOutput == Operation.INTERPOLATION:
                # Aplicamos la interpolación lineal en el DataFrame
                if axis_param == 0:
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col].dtype, np.number):
                            dataDictionary_copy[col] = dataDictionary_copy[col].apply(
                                lambda x: np.nan if get_condition(x) else x).interpolate(method='linear',
                                                                                         limit_direction='both')
                elif axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda row: row.apply(lambda x: np.nan if get_condition(x) else x).interpolate(
                            method='linear', limit_direction='both'), axis=axis_param)
                elif axis_param is None:
                    raise ValueError("The axis cannot be None when applying the INTERPOLATION operation")

            elif numOpOutput == Operation.MEAN:
                if axis_param == None:
                    # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    # Calcular la media de estas columnas numéricas
                    mean_value = only_numbers_df.mean().mean()
                    # Reemplaza 'fixValueInput' con la media del DataFrame completo usando lambda
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: mean_value if (np.issubdtype(type(x), np.number) and get_condition(x)) else x))

                elif axis_param == 0 or axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: x if not (np.issubdtype(type(x), np.number) and get_condition(x))
                            else col[col.apply(lambda z: np.issubdtype(type(z), np.number))].mean()), axis=axis_param)

            elif numOpOutput == Operation.MEDIAN:
                if axis_param == None:
                    # Seleccionar solo columnas con datos numéricos, incluyendo todos los tipos numéricos (int, float, etc.)
                    only_numbers_df = dataDictionary_copy.select_dtypes(include=[np.number])
                    # Calcular la media de estas columnas numéricas
                    median_value = only_numbers_df.median().median()
                    # Reemplaza 'fixValueInput' con la mediana del DataFrame completo usando lambda
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: median_value if (np.issubdtype(type(x), np.number) and get_condition(x)) else x))

                elif axis_param == 0 or axis_param == 1:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: x if not (np.issubdtype(type(x), np.number) and get_condition(x))
                            else col[col.apply(lambda z: np.issubdtype(type(z), np.number))].median()), axis=axis_param)

            elif numOpOutput == Operation.CLOSEST:
                if axis_param is None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(lambda x: find_closest_value(dataDictionary_copy.stack(), x)
                        if get_condition(x) else x))
                elif axis_param == 0 or axis_param == 1:
                    # Reemplazar 'fixValueInput' por el valor numérico más cercano a lo largo de las columnas
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(
                            lambda x: find_closest_value(col, x) if get_condition(x) else x), axis=axis_param)
            else:
                raise ValueError("No valid operator")

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                    if numOpOutput == Operation.INTERPOLATION:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if not get_condition(x) else np.nan).interpolate(method='linear',
                                                                                         limit_direction='both')
                    elif numOpOutput == Operation.MEAN:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if not get_condition(x) else dataDictionary_copy[field].mean())
                    elif numOpOutput == Operation.MEDIAN:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if not get_condition(x) else dataDictionary_copy[field].median())
                    elif numOpOutput == Operation.CLOSEST:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: x if not get_condition(x) else find_closest_value(dataDictionary_copy[field], x))
                else:
                    raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def checkInv_SpecialValue_FixValue(self, dataDictionary: pd.DataFrame, specialTypeInput: SpecialType,
                                       fixValueOutput, dataTypeOutput: DataType = None, missing_values: list = None,
                                       axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Check the invariant of the SpecialValue - FixValue relation
        :param dataDictionary: dataframe with the data
        :param specialTypeInput: special type of the input value
        :param dataTypeOutput: data type of the output value
        :param fixValueOutput: output value to check
        :param missing_values: list of missing values
        :param axis_param: axis to check the invariant
        :param field: field to check the invariant
        :return: dataDictionary with the values of the special type changed to the value fixValueOutput
        """
        if dataTypeOutput is not None:  # Si se especifica el tipo de dato, se realiza la transformación
            vacio, fixValueOutput = cast_type_FixValue(None, None, dataTypeOutput, fixValueOutput)

        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if specialTypeInput == SpecialType.MISSING:  # Incluye nulos, np.nan, None, etc y los valores de la lista missing_values
                dataDictionary_copy = dataDictionary_copy.replace(np.nan, fixValueOutput)
                if missing_values is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))

            elif specialTypeInput == SpecialType.INVALID:  # Solo incluye los valores de la lista missing_values
                if missing_values is not None:
                    dataDictionary_copy = dataDictionary_copy.apply(
                        lambda col: col.apply(lambda x: fixValueOutput if x in missing_values else x))

            elif specialTypeInput == SpecialType.OUTLIER:#TODO:Probar
                threshold = 1.5
                if axis_param is None:
                    Q1 = dataDictionary_copy.stack().quantile(0.25)
                    Q3 = dataDictionary_copy.stack().quantile(0.75)
                    IQR = Q3 - Q1
                    # Definir los límites para identificar outliers
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    # Identificar outliers en el dataframe completo
                    numeric_values = dataDictionary_copy.select_dtypes(include=[np.number])
                    outliers = (numeric_values < lower_bound) | (numeric_values > upper_bound)
                    # Reemplazar outliers por fixValueOutput
                    dataDictionary_copy[outliers] = fixValueOutput

                elif axis_param == 0:#Se niega la condición para que se cumpla la condición de outliers, y se reemplaza por fixValueOutput en el else
                    for col in dataDictionary_copy.columns:
                        if np.issubdtype(dataDictionary_copy[col], np.number):
                            dataDictionary_copy[col] = dataDictionary_copy[col].where(~((
                                        dataDictionary_copy[col] < dataDictionary_copy[col].quantile(0.25) - threshold * (
                                        dataDictionary_copy[col].quantile(0.75) - dataDictionary_copy[col].quantile(0.25))) |
                                        (dataDictionary_copy[col] > dataDictionary_copy[col].quantile(0.75) + threshold * (
                                        dataDictionary_copy[col].quantile(0.75) - dataDictionary_copy[col].quantile(0.25)))),
                                                                                other = fixValueOutput)

                elif axis_param == 1:
                    Q1 = dataDictionary_copy.quantile(0.25, axis="rows")
                    Q3 = dataDictionary_copy.quantile(0.75, axis="rows")
                    IQR = Q3 - Q1
                    outliers = dataDictionary_copy[
                        (dataDictionary_copy < Q1 - threshold * IQR) | (dataDictionary_copy > Q3 + threshold * IQR)]
                    for row in outliers.index:
                        dataDictionary_copy.iloc[row] = dataDictionary_copy.iloc[row].where(
                            ~((dataDictionary_copy.iloc[row] < Q1.iloc[row] - threshold * IQR.iloc[row]) |
                              (dataDictionary_copy.iloc[row] > Q3.iloc[row] + threshold * IQR.iloc[row])),
                                other=fixValueOutput)

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")

            elif field in dataDictionary.columns:
                if specialTypeInput == SpecialType.MISSING:
                    dataDictionary_copy[field] = dataDictionary_copy[field].replace(np.nan, fixValueOutput)
                    if missing_values is not None:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: fixValueOutput if x in missing_values else x)
                elif specialTypeInput == SpecialType.INVALID:
                    if missing_values is not None:
                        dataDictionary_copy[field] = dataDictionary_copy[field].apply(
                            lambda x: fixValueOutput if x in missing_values else x)
                elif specialTypeInput == SpecialType.OUTLIER:
                    if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                        threshold = 1.5
                        Q1 = dataDictionary_copy[field].quantile(0.25)
                        Q3 = dataDictionary_copy[field].quantile(0.75)
                        IQR = Q3 - Q1
                        dataDictionary_copy[field] = dataDictionary_copy[field].where(
                            ~((dataDictionary_copy[field] < Q1 - threshold * IQR) |
                              (dataDictionary_copy[field] > Q3 + threshold * IQR)),
                            other=fixValueOutput)
                    else:
                        raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def checkInv_SpecialValue_DerivedValue(self, dataDictionary: pd.DataFrame, specialTypeInput: SpecialType,
                                           derivedTypeOutput: DerivedType, missing_values: list = None,
                                           axis_param: int = None, field: str = None) -> pd.DataFrame:
        """
        Check the invariant of the SpecialValue - DerivedValue relation
        :param dataDictionary: dataframe with the data
        :param specialTypeInput: special type of the input value
        :param derivedTypeOutput: derived type of the output value
        :param missing_values: list of missing values
        :param axis_param: axis to check the invariant
        :param field: field to check the invariant
        :return: dataDictionary with the values of the special type changed to the value derived from the operation derivedTypeOutput
        """
        dataDictionary_copy = dataDictionary.copy()

        if field is None:
            if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
                dataDictionary_copy = apply_derivedType(specialTypeInput, derivedTypeOutput, dataDictionary_copy,
                                                        missing_values, axis_param, field)

            elif specialTypeInput == SpecialType.OUTLIER:
                # IMPORTANTE: El valor de axis_param que se aplica a la función getOutliers() es el mismo qu el que se utiliza
                # en la función apply_derivedTypeOutliers(). Por tanto, si se aplican los OUTLIERS a nivel de dataframe, no se podrá aplicar
                # ni previous ni next.

                if axis_param is None:
                    dataDictionary_copy_copy = getOutliers(dataDictionary_copy, field, axis_param)
                    missing_values = dataDictionary_copy.where(dataDictionary_copy_copy == 1).stack().tolist()
                    dataDictionary_copy = apply_derivedType(specialTypeInput, derivedTypeOutput, dataDictionary_copy,
                                                            missing_values, axis_param, field)
                elif axis_param == 0 or axis_param == 1:
                    dataDictionary_copy_copy = getOutliers(dataDictionary_copy, field, axis_param)
                    dataDictionary_copy = apply_derivedTypeColRowOutliers(derivedTypeOutput, dataDictionary_copy,
                                                                          dataDictionary_copy_copy, axis_param, field)

        elif field is not None:
            if field not in dataDictionary.columns:
                raise ValueError("The field does not exist in the dataframe")
            elif field in dataDictionary.columns:
                if np.issubdtype(dataDictionary_copy[field].dtype, np.number):
                    if specialTypeInput == SpecialType.MISSING or specialTypeInput == SpecialType.INVALID:
                        dataDictionary_copy = apply_derivedType(specialTypeInput, derivedTypeOutput, dataDictionary_copy,
                                                                missing_values, axis_param, field)
                    elif specialTypeInput == SpecialType.OUTLIER:
                        dataDictionary_copy_copy = getOutliers(dataDictionary_copy, field, axis_param)
                        dataDictionary_copy = apply_derivedTypeColRowOutliers(derivedTypeOutput, dataDictionary_copy,
                                                                              dataDictionary_copy_copy, axis_param, field)
                else:
                    raise ValueError("The field is not numeric")

        return dataDictionary_copy

    def checkInv_SpecialValue_NumOp(self, dataDictionary: pd.DataFrame, specialTypeInput: SpecialType,
                                    numOpOutput: Operation,
                                    missing_values: list = None, axis_param: int = None,
                                    field: str = None) -> pd.DataFrame:
        """
        Check the invariant of the SpecialValue - NumOp relation
        :param dataDictionary: dataframe with the data
        :param specialTypeInput: special type of the input value
        :param numOpOutput: operation to check the invariant
        :param axis_param: axis to check the invariant
        :param field: field to check the invariant
        :return: dataDictionary with the values of the special type changed to the result of the operation numOpOutput
        """
        dataDictionary_copy = dataDictionary.copy()
        dataDictionary_copy_mask = None

        if specialTypeInput == SpecialType.OUTLIER:
            dataDictionary_copy_mask = getOutliers(dataDictionary_copy, field, axis_param)

        if numOpOutput == Operation.INTERPOLATION:
            dataDictionary_copy = specialTypeInterpolation(dataDictionary_copy=dataDictionary_copy,
                                                           specialTypeInput=specialTypeInput,
                                                           dataDictionary_copy_mask=dataDictionary_copy_mask,
                                                           missing_values=missing_values, axis_param=axis_param,
                                                           field=field)
        elif numOpOutput == Operation.MEAN:
            dataDictionary_copy = specialTypeMean(dataDictionary_copy=dataDictionary_copy,
                                                  specialTypeInput=specialTypeInput,
                                                  dataDictionary_copy_mask=dataDictionary_copy_mask,
                                                  missing_values=missing_values, axis_param=axis_param, field=field)
        elif numOpOutput == Operation.MEDIAN:
            dataDictionary_copy = specialTypeMedian(dataDictionary_copy=dataDictionary_copy,
                                                    specialTypeInput=specialTypeInput,
                                                    dataDictionary_copy_mask=dataDictionary_copy_mask,
                                                    missing_values=missing_values, axis_param=axis_param, field=field)
        elif numOpOutput == Operation.CLOSEST:
            dataDictionary_copy = specialTypeClosest(dataDictionary_copy=dataDictionary_copy,
                                                     specialTypeInput=specialTypeInput,
                                                     dataDictionary_copy_mask=dataDictionary_copy_mask,
                                                     missing_values=missing_values, axis_param=axis_param, field=field)

        return dataDictionary_copy
