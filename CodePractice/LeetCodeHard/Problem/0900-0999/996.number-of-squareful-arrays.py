"""
LC 996. Number of Squareful Arrays — Hard
https://leetcode.com/problems/number-of-squareful-arrays/
Best: Graph + backtrack / Hamilton paths with dup — O(n! * n)
"""

from typing import List
from collections import Counter
import math


class Solution:
    def numSquarefulPerms(self, nums: List[int]) -> int:
        def is_sq(x):
            r = int(math.isqrt(x))
            return r * r == x

        n = len(nums)
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and is_sq(nums[i] + nums[j]):
                    graph[i].append(j)

        self.ans = 0

        def dfs(u, used, left):
            if left == 0:
                self.ans += 1
                return
            for v in graph[u]:
                if used & (1 << v):
                    continue
                # skip duplicate values at same depth
                if any(nums[v] == nums[w] and (used & (1 << w)) == 0 and w < v for w in range(n)):
                    continue
                dfs(v, used | (1 << v), left - 1)

        # start from each unique first value
        for i in range(n):
            if any(nums[i] == nums[j] and j < i for j in range(n)):
                continue
            dfs(i, 1 << i, n - 1)
        return self.ans


def _test():
    s = Solution()
    assert s.numSquarefulPerms([1, 17, 8]) == 2
    assert s.numSquarefulPerms([2, 2, 2]) == 1
    print("LC 996 OK")


if __name__ == "__main__":
    _test()
