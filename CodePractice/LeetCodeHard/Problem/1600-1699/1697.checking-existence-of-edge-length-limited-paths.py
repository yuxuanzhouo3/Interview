"""
LC 1697. Checking Existence of Edge Length Limited Paths — Hard
https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/
Best: sort queries + Union-Find offline — O((n+q) α(n) + m log m)
"""

from typing import List


class Solution:
    def distanceLimitedPathsExist(
        self, n: int, edgeList: List[List[int]], queries: List[List[int]]
    ) -> List[bool]:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        edgeList.sort(key=lambda e: e[2])
        qi = sorted(range(len(queries)), key=lambda i: queries[i][2])
        ans = [False] * len(queries)
        j = 0
        for i in qi:
            p, q, lim = queries[i]
            while j < len(edgeList) and edgeList[j][2] < lim:
                union(edgeList[j][0], edgeList[j][1])
                j += 1
            ans[i] = find(p) == find(q)
        return ans


def _test():
    s = Solution()
    assert s.distanceLimitedPathsExist(
        3, [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 16]], [[0, 1, 2], [0, 2, 5]]
    ) == [False, True]
    assert s.distanceLimitedPathsExist(
        5,
        [[0, 1, 10], [1, 2, 5], [2, 3, 9], [3, 4, 13]],
        [[0, 4, 14], [1, 4, 13]],
    ) == [True, False]
    print("LC 1697 OK")


if __name__ == "__main__":
    _test()
