"""
LC 3695. Maximize Alternating Sum Using Swaps — Hard
https://leetcode.com/problems/maximize-alternating-sum-using-swaps/
Best: Union-Find components + sort assign — O(n log n)
"""

from typing import List


class Solution:
    def maxAlternatingSum(self, nums: List[int], swaps: List[List[int]]) -> int:
        n = len(nums)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        for a, b in swaps:
            union(a, b)

        from collections import defaultdict

        comps = defaultdict(list)
        for i in range(n):
            comps[find(i)].append(i)

        ans = 0
        for idxs in comps.values():
            vals = sorted((nums[i] for i in idxs), reverse=True)
            # even indices want positive contribution, odd negative in alternating sum
            # within component, we can reassign values to positions freely
            even = sum(1 for i in idxs if i % 2 == 0)
            # assign largest `even` values to even positions
            for i, v in enumerate(vals):
                if i < even:
                    ans += v
                else:
                    ans -= v
        return ans


def _test():
    s = Solution()
    assert s.maxAlternatingSum([1, 2, 3], [[0, 2], [1, 2]]) == 4
    assert s.maxAlternatingSum([1, 2, 3], [[1, 2]]) == 2
    print("LC 3695 OK")


if __name__ == "__main__":
    _test()
