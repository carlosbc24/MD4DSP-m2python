
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


class DataType(Enum):
    """
    Enumeration for the data type

    STRING: String
    TIME: Time
    INTEGER: Integer
    DATETIME: Datetime
    BOOLEAN: Boolean
    DOUBLE: Double
    FLOAT: Float
    """
    STRING = 0
    TIME = 1
    INTEGER = 2
    DATETIME = 3
    BOOLEAN = 4
    DOUBLE = 5
    FLOAT = 6


class DerivedType(Enum):
    """
    Enumeration for the derived type

    MOSTFREQUENT: Most frequent value
    PREVIOUS: Previous value
    NEXT: Next value
    """
    MOSTFREQUENT = 0
    PREVIOUS = 1
    NEXT = 2


class Operation(Enum):
    """
    Enumeration for the operation

    INTERPOLATION: Interpolation
    MEAN: Mean
    MEDIAN: Median
    CLOSEST: Closest
    """
    INTERPOLATION = 0
    MEAN = 1
    MEDIAN = 2
    CLOSEST = 3


class SpecialType(Enum):
    """
    Enumeration for the special type

    MISSING: Missing value
    INVALID: Invalid value
    OUTLIER: Outlier value
    """
    MISSING = 0
    INVALID = 1
    OUTLIER = 2
