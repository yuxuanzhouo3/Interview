"""
LC 2307. Check for Contradictions in Equations — Hard
https://leetcode.com/problems/check-for-contradictions-in-equations/
Best: Weighted Union-Find (ratio) — O(n α(n))
"""
from typing import List


class Solution:
    def checkContradictions(self, equations: List[List[str]], values: List[float]) -> bool:
        parent = {}
        weight = {}  # x / parent(x) = weight[x]

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
                weight[x] = 1.0
                return x
            if parent[x] != x:
                p = find(parent[x])
                weight[x] *= weight[parent[x]]
                parent[x] = p
            return parent[x]

        EPS = 1e-5
        for (a, b), v in zip(equations, values):
            ra, rb = find(a), find(b)
            if ra == rb:
                if abs(weight[a] / weight[b] - v) > EPS:
                    return True
            else:
                parent[ra] = rb
                weight[ra] = v * weight[b] / weight[a]
        return False


def _test():
    s = Solution()
    assert s.checkContradictions([["a", "b"], ["b", "c"], ["a", "c"]], [3.0, 0.5, 1.4]) is True
    # 2 * (1/5) would need code/et = 0.4 for consistency; 0.5 contradicts
    assert s.checkContradictions([["le", "et"], ["le", "code"], ["code", "et"]], [2.0, 5.0, 0.5]) is True
    assert s.checkContradictions([["le", "et"], ["le", "code"], ["code", "et"]], [2.0, 5.0, 0.4]) is False
    assert s.checkContradictions([["a", "b"], ["b", "c"], ["a", "c"]], [3.0, 0.5, 1.5]) is False
    print("LC 2307 OK")


if __name__ == "__main__":
    _test()
