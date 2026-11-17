from functools import cache
from bisect import bisect_left, bisect_right
"""
LC 3661. Maximum Walls Destroyed by Robots — Hard
https://leetcode.com/problems/maximum-walls-destroyed-by-robots/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def maxWalls(self, robots: list[int], distance: list[int], walls: list[int]) -> int:
        n = len(robots)
        arr = sorted(zip(robots, distance), key=lambda x: x[0])
        walls.sort()

        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0
            left = arr[i][0] - arr[i][1]
            if i > 0:
                left = max(left, arr[i - 1][0] + 1)
            l = bisect_left(walls, left)
            r = bisect_left(walls, arr[i][0] + 1)
            ans = dfs(i - 1, 0) + r - l
            right = arr[i][0] + arr[i][1]
            if i + 1 < n:
                if j == 0:
                    right = min(right, arr[i + 1][0] - arr[i + 1][1] - 1)
                else:
                    right = min(right, arr[i + 1][0] - 1)
            l = bisect_left(walls, arr[i][0])
            r = bisect_left(walls, right + 1)
            ans = max(ans, dfs(i - 1, 1) + r - l)
            return ans

        ans = dfs(n - 1, 1)
        dfs.cache_clear()
        return ans

def _test():
    s = Solution()
    assert s.maxWalls([4], [3], [1,10]) == 1
    assert s.maxWalls([10,2], [5,1], [5,2,7]) == 3
    assert s.maxWalls([1,2], [100,1], [10]) == 0
    print("LC 3661 OK")


if __name__ == "__main__":
    _test()
