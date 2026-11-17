"""
LC 1483. Kth Ancestor of a Tree Node — Hard
https://leetcode.com/problems/kth-ancestor-of-a-tree-node/
Best: Binary lifting — O(n log n) build, O(log n) query
"""

from typing import List


class TreeAncestor:
    def __init__(self, n: int, parent: List[int]):
        self.LOG = 16  # 2^16 > 5e4
        self.up = [[-1] * n for _ in range(self.LOG)]
        for i in range(n):
            self.up[0][i] = parent[i]
        for j in range(1, self.LOG):
            for i in range(n):
                p = self.up[j - 1][i]
                self.up[j][i] = -1 if p == -1 else self.up[j - 1][p]

    def getKthAncestor(self, node: int, k: int) -> int:
        for j in range(self.LOG):
            if k & (1 << j):
                node = self.up[j][node]
                if node == -1:
                    return -1
        return node


def _test():
    t = TreeAncestor(7, [-1, 0, 0, 1, 1, 2, 2])
    assert t.getKthAncestor(3, 1) == 1
    assert t.getKthAncestor(5, 2) == 0
    assert t.getKthAncestor(6, 3) == -1
    print("LC 1483 OK")


if __name__ == "__main__":
    _test()
