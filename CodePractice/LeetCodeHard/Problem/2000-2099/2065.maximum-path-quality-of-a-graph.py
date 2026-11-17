"""
LC 2065. Maximum Path Quality of a Graph — Hard
https://leetcode.com/problems/maximum-path-quality-of-a-graph/
Best: DFS backtrack with time limit — O(branching^depth), n<=22
"""

from typing import List


class Solution:
    def maximalPathQuality(self, values: List[int], edges: List[List[int]], maxTime: int) -> int:
        n = len(values)
        g: List[List[tuple]] = [[] for _ in range(n)]
        for u, v, t in edges:
            g[u].append((v, t))
            g[v].append((u, t))

        ans = 0

        def dfs(u: int, time: int, quality: int, vis: set) -> None:
            nonlocal ans
            if u == 0:
                ans = max(ans, quality)
            for v, t in g[u]:
                nt = time + t
                if nt > maxTime:
                    continue
                if v in vis:
                    dfs(v, nt, quality, vis)
                else:
                    vis.add(v)
                    dfs(v, nt, quality + values[v], vis)
                    vis.remove(v)

        dfs(0, 0, values[0], {0})
        return ans


def _test():
    s = Solution()
    assert s.maximalPathQuality([0, 32, 10, 43], [[0, 1, 10], [1, 2, 15], [0, 3, 10]], 49) == 75
    assert s.maximalPathQuality([5, 10, 15, 20], [[0, 1, 10], [1, 2, 10], [0, 3, 10]], 30) == 25
    assert s.maximalPathQuality([1, 2, 3, 4], [[0, 1, 10], [1, 2, 11], [2, 3, 12], [1, 3, 13]], 50) == 7
    print("LC 2065 OK")


if __name__ == "__main__":
    _test()
