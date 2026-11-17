"""
LC 2440. Create Components With Same Value — Hard
https://leetcode.com/problems/create-components-with-same-value/
Best: Enumerate component sum divisors + DFS — O(n · d(S))
"""

from collections import defaultdict
from typing import List

class Solution:
    def componentValue(self, nums: List[int], edges: List[List[int]]) -> int:
        def dfs(i, fa):
            x = nums[i]
            for j in g[i]:
                if j != fa:
                    y = dfs(j, i)
                    if y == -1:
                        return -1
                    x += y
            if x > t:
                return -1
            return x if x < t else 0

        n = len(nums)
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        s = sum(nums)
        mx = max(nums)
        for k in range(min(n, s // mx), 1, -1):
            if s % k == 0:
                t = s // k
                if dfs(0, -1) == 0:
                    return k - 1
        return 0

def _test():

    s = Solution()
    assert s.componentValue([6, 2, 2, 2, 6], [[0, 1], [1, 2], [1, 3], [3, 4]]) == 2
    assert s.componentValue([2], []) == 0
    print("LC 2440 OK")

if __name__ == "__main__":
    _test()
