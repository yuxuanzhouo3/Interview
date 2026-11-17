"""
LC 3244. Shortest Distance After Road Addition Queries II — Hard
https://leetcode.com/problems/shortest-distance-after-road-addition-queries-ii/
Best: Next-pointer cover, erase intermediates — O(n+q)
"""

from typing import List


class Solution:
    def shortestDistanceAfterQueries(
        self, n: int, queries: List[List[int]]
    ) -> List[int]:
        nxt = list(range(1, n)) + [n]
        cur = n - 1
        ans = []
        for u, v in queries:
            if nxt[u] >= v:
                ans.append(cur)
                continue
            i = nxt[u]
            while i < v:
                ni = nxt[i]
                nxt[i] = n
                cur -= 1
                i = ni
            nxt[u] = v
            ans.append(cur)
        return ans


def _test():
    s = Solution()
    assert s.shortestDistanceAfterQueries(5, [[2, 4], [0, 2], [0, 4]]) == [3, 2, 1]
    assert s.shortestDistanceAfterQueries(4, [[0, 3], [0, 2]]) == [1, 1]
    print("LC 3244 OK")


if __name__ == "__main__":
    _test()
