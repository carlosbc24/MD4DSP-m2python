from enum import Enum
import pandas as pd


class Belong(Enum):
    BELONG = 0
    NOTBELONG = 1


class ContractsPrePost:
    def checkFieldRange(self, fields: list, dataDictionary: pd.DataFrame, belongOp: Belong) -> bool:
        """
        Check if fields meets the condition of belongOp in dataDictionary

        :param fields:
        :param:dataDictionary:
        :param:belongOp:

        :return: if fields meets the condition of belongOp in dataDictionary
        :rtype: bool
        """
        return False


if __name__ == '__main__':
    pre_post = ContractsPrePost()

    ## Case 1 of checkFieldRange
    fields = ['c1', 'c2']
    dataDictionary = pd.DataFrame(columns=['c1', 'c2'])
    belong = 0
    pre_post.checkFieldRange(fields=fields, dataDictionary=dataDictionary, belongOp=Belong(belong))

