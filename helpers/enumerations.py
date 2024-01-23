from enum import Enum


class Belong(Enum):
    BELONG = 0
    NOTBELONG = 1


class Operator(Enum):
    GREATEREQUAL = 0
    GREATER = 1
    LESSEQUAL = 2
    LESS = 3
    EQUAL = 4