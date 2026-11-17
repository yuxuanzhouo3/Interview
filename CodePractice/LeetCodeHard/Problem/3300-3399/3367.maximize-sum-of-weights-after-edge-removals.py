"""
LC 3367. Maximize Sum of Weights after Edge Removals — Hard
https://leetcode.com/problems/maximize-sum-of-weights-after-edge-removals/
Best: 方法一
"""

from typing import Tuple

class Solution:
    def maximizeSumOfWeights(self, edges: list[list[int]], k: int) -> int:
        def dfs(u: int, fa: int) -> Tuple[int, int]:
            s = 0
            t = []
            for v, w in g[u]:
                if v == fa:
                    continue
                a, b = dfs(v, u)
                s += a
                if (d := (w + b - a)) > 0:
                    t.append(d)
            t.sort(reverse=True)
            return s + sum(t[:k]), s + sum(t[: k - 1])

        n = len(edges) + 1
        g: list[list[Tuple[int, int]]] = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        x, y = dfs(0, -1)
        return max(x, y)

def _test():
    s = Solution()
    assert s.maximizeSumOfWeights(edges=[[0, 1, 4], [0, 2, 2], [2, 3, 12], [2, 4, 6]], k=2) == 22
    assert s.maximizeSumOfWeights(edges=[[0, 1, 5], [1, 2, 10], [0, 3, 15], [3, 4, 20], [3, 5, 5], [0, 6, 10]], k=3) == 65
    print("LC 3367 OK")

if __name__ == "__main__":
    _test()
