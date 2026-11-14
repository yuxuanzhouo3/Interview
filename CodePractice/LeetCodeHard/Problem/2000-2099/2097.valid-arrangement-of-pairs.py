"""
LC 2097. Valid Arrangement of Pairs — Hard
https://leetcode.com/problems/valid-arrangement-of-pairs/
Best: Hierholzer Euler path — O(E)
"""

from typing import List
from collections import defaultdict, deque


class Solution:
    def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
        g = defaultdict(deque)
        din = defaultdict(int)
        dout = defaultdict(int)
        for a, b in pairs:
            g[a].append(b)
            dout[a] += 1
            din[b] += 1
        start = pairs[0][0]
        for u in set(dout) | set(din):
            if dout[u] - din[u] == 1:
                start = u
                break
        path = []

        def dfs(u):
            while g[u]:
                v = g[u].popleft()
                dfs(v)
                path.append([u, v])

        dfs(start)
        path.reverse()
        return path


def _test():
    s = Solution()
    r = s.validArrangement([[5, 1], [4, 5], [11, 9], [9, 4]])
    assert len(r) == 4
    for i in range(len(r) - 1):
        assert r[i][1] == r[i + 1][0]
    r2 = s.validArrangement([[1, 3], [3, 2], [2, 1]])
    assert len(r2) == 3
    print("LC 2097 OK")


if __name__ == "__main__":
    _test()
