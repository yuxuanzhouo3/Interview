"""
LC 1340. Jump Game V — Hard
https://leetcode.com/problems/jump-game-v/
Best: DFS + memo from each index — O(n d)
"""

from typing import List
from functools import cache


class Solution:
    def maxJumps(self, arr: List[int], d: int) -> int:
        n = len(arr)

        @cache
        def dfs(i: int) -> int:
            best = 1
            for direction in (-1, 1):
                for step in range(1, d + 1):
                    j = i + direction * step
                    if j < 0 or j >= n or arr[j] >= arr[i]:
                        break
                    best = max(best, 1 + dfs(j))
            return best

        return max(dfs(i) for i in range(n))


def _test():
    s = Solution()
    assert s.maxJumps([6, 4, 14, 6, 8, 13, 9, 7, 10, 6, 12], 2) == 4
    assert s.maxJumps([3, 3, 3, 3, 3], 3) == 1
    assert s.maxJumps([7, 6, 5, 4, 3, 2, 1], 1) == 7
    print("LC 1340 OK")


if __name__ == "__main__":
    _test()
