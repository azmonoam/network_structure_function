from typing import Union

Logical = Union[bool, int]


class LogicalGates:

    @staticmethod
    def AND(
            a: Logical,
            b: Logical,
    ) -> Logical:
        return a & b

    @staticmethod
    def OR(
            a: Logical,
            b: Logical,
    ) -> Logical:
        return a | b

    @staticmethod
    def NOT(
            a: Logical,
    ) -> Logical:
        return ~a + 2

    def NAND(
            self,
            a: Logical,
            b: Logical,
    ) -> Logical:
        return self.NOT(self.AND(a, b))

    @staticmethod
    def XOR(
            a: Logical,
            b: Logical,
    ) -> Logical:
        return a ^ b
