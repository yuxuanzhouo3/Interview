"""
LC 2959. Number of Possible Sets of Closing Branches — Hard
https://leetcode.com/problems/number-of-possible-sets-of-closing-branches/
Best: Enumerate subsets + Floyd — O(2^n * n^3)
"""

from typing import List


class Solution:
    def numberOfSets(self, n: int, maxDistance: int, roads: List[List[int]]) -> int:
        INF = 10**9
        ans = 0
        for mask in range(1 << n):
            dist = [[INF] * n for _ in range(n)]
            for i in range(n):
                if mask >> i & 1:
                    dist[i][i] = 0
            for u, v, w in roads:
                if (mask >> u & 1) and (mask >> v & 1):
                    dist[u][v] = min(dist[u][v], w)
                    dist[v][u] = min(dist[v][u], w)
            for k in range(n):
                if not (mask >> k & 1):
                    continue
                for i in range(n):
                    if not (mask >> i & 1):
                        continue
                    for j in range(n):
                        if not (mask >> j & 1):
                            continue
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
            ok = True
            nodes = [i for i in range(n) if mask >> i & 1]
            for i in nodes:
                for j in nodes:
                    if dist[i][j] > maxDistance:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                ans += 1
        return ans


def _test():
    s = Solution()
    assert s.numberOfSets(3, 5, [[0, 1, 2], [1, 2, 10], [0, 2, 10]]) == 5
    assert s.numberOfSets(3, 5, [[0, 1, 20], [0, 1, 10], [1, 2, 2], [0, 2, 2]]) == 7
    assert s.numberOfSets(1, 10, []) == 2
    print("LC 2959 OK")


if __name__ == "__main__":
    _test()
