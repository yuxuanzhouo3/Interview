"""
LC 631. Design Excel Sum Formula — Hard
https://leetcode.com/problems/design-excel-sum-formula/
Tags: Design, Graph
Best: 格子存值/公式，求值时递归求和依赖
"""

from typing import List
from collections import defaultdict


class Excel:
    def __init__(self, height: int, width: str):
        self.h = height
        self.w = ord(width) - ord('A') + 1
        self.val = [[0] * self.w for _ in range(self.h)]
        self.formula = {}  # (r,c) -> list of (r,c) cells counted with multiplicity

    def set(self, row: int, column: str, val: int) -> None:
        r, c = row - 1, ord(column) - ord('A')
        self.formula.pop((r, c), None)
        self.val[r][c] = val

    def get(self, row: int, column: str) -> int:
        r, c = row - 1, ord(column) - ord('A')
        return self._eval(r, c)

    def sum(self, row: int, column: str, numbers: List[str]) -> int:
        r, c = row - 1, ord(column) - ord('A')
        cells = []
        for token in numbers:
            if ':' in token:
                a, b = token.split(':')
                r1, c1 = int(a[1:]) - 1, ord(a[0]) - ord('A')
                r2, c2 = int(b[1:]) - 1, ord(b[0]) - ord('A')
                for i in range(r1, r2 + 1):
                    for j in range(c1, c2 + 1):
                        cells.append((i, j))
            else:
                cells.append((int(token[1:]) - 1, ord(token[0]) - ord('A')))
        self.formula[(r, c)] = cells
        return self._eval(r, c)

    def _eval(self, r, c):
        if (r, c) not in self.formula:
            return self.val[r][c]
        return sum(self._eval(i, j) for i, j in self.formula[(r, c)])


def _test():
    excel = Excel(3, 'C')
    excel.set(1, 'A', 2)
    assert excel.sum(3, 'C', ['A1', 'A1:B2']) == 4
    excel.set(2, 'B', 2)
    assert excel.get(3, 'C') == 6
    print("LC 631 OK")


if __name__ == "__main__":
    _test()

