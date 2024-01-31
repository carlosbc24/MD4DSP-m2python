from typing import Union

import pandas as pd

from helpers.enumerations import Operator


def compare_numbers(rel_number: Union [int,float], quant_rel: float, quant_op: Operator) -> bool:
    """
    Compare two numbers with the operator quant_op

    :param rel_number: relative number to compare
    :param quant_rel: relative number to compare with the previous one
    :param quant_op: quantifier operator to compare the two numbers

    :return: if rel_number meets the condition of quant_op with quant_rel
    """
    if quant_op == Operator.GREATEREQUAL:
        return rel_number >= quant_rel
    elif quant_op == Operator.GREATER:
        return rel_number > quant_rel
    elif quant_op == Operator.LESSEQUAL:
        return rel_number <= quant_rel
    elif quant_op == Operator.LESS:
        return rel_number < quant_rel
    elif quant_op == Operator.EQUAL:
        return rel_number == quant_rel
    else:
        raise ValueError("No valid operator")


def count_abs_frequency(value, dataDictionary: pd.DataFrame)->int:
    """
    Count the frequency of a value in all the columns of a dataframe

    :param value: value to count
    :param dataDictionary: dataframe with the data

    :return: frequency of the value in the column
    """
    count = 0
    for column in dataDictionary:
        count += dataDictionary[column].value_counts(dropna=False).get(value, 0)
    return count