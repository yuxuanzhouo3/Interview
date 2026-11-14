"""
LC 952. Largest Component Size by Common Factor — Hard
https://leetcode.com/problems/largest-component-size-by-common-factor/
Best: Union-Find on prime factors — O(n * sqrt(A))
"""

from typing import List
from collections import Counter


class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        m = max(nums)
        parent = list(range(m + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        def factors(x):
            res = []
            d = 2
            while d * d <= x:
                if x % d == 0:
                    res.append(d)
                    while x % d == 0:
                        x //= d
                d += 1
            if x > 1:
                res.append(x)
            return res

        for x in nums:
            for f in factors(x):
                union(x, f)

        cnt = Counter(find(x) for x in nums)
        return max(cnt.values())


def _test():
    s = Solution()
    assert s.largestComponentSize([4, 6, 15, 35]) == 4
    assert s.largestComponentSize([20, 50, 9, 63]) == 2
    assert s.largestComponentSize([2, 3, 6, 7, 4, 12, 21, 39]) == 8
    print("LC 952 OK")


if __name__ == "__main__":
    _test()
