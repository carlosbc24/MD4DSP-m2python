from helpers.enumerations import Operator


def comparar_numeros(rel_number, quant_rel, quant_op) -> bool:
    '''
    Compare two numbers with the operator quant_op

    :param rel_number: relative number to compare
    :param quant_rel: relative number to compare with the previous one
    :param quant_op: quantifier operator to compare the two numbers

    :return: if rel_number meets the condition of quant_op with quant_rel
    '''
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