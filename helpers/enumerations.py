
# Importing libraries
from enum import Enum


class Belong(Enum):
    """
    Enumeration for the belong relation

    BELONG: The element belongs to the set
    NOTBELONG: The element does not belong to the set
    """
    BELONG = 0
    NOTBELONG = 1


class Operator(Enum):
    """
    Enumeration for the quantifier operators

    GREATEREQUAL: Greater or equal
    GREATER: Greater
    LESSEQUAL: Less or equal
    LESS: Less
    EQUAL: Equal
    """
    GREATEREQUAL = 0
    GREATER = 1
    LESSEQUAL = 2
    LESS = 3
    EQUAL = 4


class Closure(Enum):
    """
    Enumeration for the closure of the interval

    openOpen: open interval
    openClosed: open left and closed right
    closedOpen: closed left and open right
    closedClosed: closed interval
    """
    openOpen = 0
    openClosed = 1
    closedOpen = 2
    closedClosed = 3
