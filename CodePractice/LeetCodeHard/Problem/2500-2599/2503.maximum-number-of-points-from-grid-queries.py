"""
LC 2503. Maximum Number of Points From Grid Queries — Hard
https://leetcode.com/problems/maximum-number-of-points-from-grid-queries/
Best: Sort queries + min-heap expand reachable cells — O(mn log mn + q log q)
"""

from typing import List
import heapq


class Solution:
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        m, n = len(grid), len(grid[0])
        order = sorted(range(len(queries)), key=lambda i: queries[i])
        ans = [0] * len(queries)
        heap = [(grid[0][0], 0, 0)]
        seen = {(0, 0)}
        points = 0
        for qi in order:
            q = queries[qi]
            while heap and heap[0][0] < q:
                _, i, j = heapq.heappop(heap)
                points += 1
                for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen:
                        seen.add((ni, nj))
                        heapq.heappush(heap, (grid[ni][nj], ni, nj))
            ans[qi] = points
        return ans


def _test():
    s = Solution()
    assert s.maxPoints([[1, 2, 3], [2, 5, 7], [3, 5, 1]], [5, 6, 2]) == [5, 8, 1]
    assert s.maxPoints([[5, 2, 1], [1, 1, 2]], [3]) == [0]
    print("LC 2503 OK")


if __name__ == "__main__":
    _test()
