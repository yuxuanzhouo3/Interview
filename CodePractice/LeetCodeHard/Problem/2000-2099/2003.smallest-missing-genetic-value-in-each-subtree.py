"""
LC 2003. Smallest Missing Genetic Value in Each Subtree — Hard
https://leetcode.com/problems/smallest-missing-genetic-value-in-each-subtree/
Best: DFS collect mex; special path from node with 1 — O(n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def smallestMissingValueSubtree(self, parents: List[int], nums: List[int]) -> List[int]:
        n = len(parents)
        g = defaultdict(list)
        for i, p in enumerate(parents):
            if p >= 0:
                g[p].append(i)
        ans = [1] * n
        if 1 not in nums:
            return ans
        root1 = nums.index(1)
        seen = set()
        mex = 1

        def dfs_collect(u: int) -> None:
            seen.add(nums[u])
            for v in g[u]:
                dfs_collect(v)

        u = root1
        prev = -1
        while u != -1:
            for v in g[u]:
                if v != prev:
                    dfs_collect(v)
            seen.add(nums[u])
            while mex in seen:
                mex += 1
            ans[u] = mex
            prev = u
            u = parents[u]
        return ans


def _test():
    s = Solution()
    assert s.smallestMissingValueSubtree([-1, 0, 0, 2], [1, 2, 3, 4]) == [5, 1, 1, 1]
    assert s.smallestMissingValueSubtree([-1, 0, 1, 0, 3, 3], [5, 4, 6, 2, 1, 3]) == [7, 1, 1, 4, 2, 1]
    assert s.smallestMissingValueSubtree([-1, 2, 3, 0, 2, 4, 1], [2, 3, 4, 5, 6, 7, 8]) == [1, 1, 1, 1, 1, 1, 1]
    print("LC 2003 OK")


if __name__ == "__main__":
    _test()
