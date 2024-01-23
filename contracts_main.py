from helpers.logger import set_logger
from test.simple_test import ContractTest

if __name__ == "__main__":

    set_logger("test")

    contract_test = ContractTest()
    contract_test.execute_CheckFieldRange_Tests()
    contract_test.execute_CheckFixValueRangeString_Tests()
