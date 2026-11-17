"""
LC 2519. Count the Number of K-Big Indices — Hard
Best: Fenwick both directions — O(n log n)
"""

from typing import List


class BIT:
    def __init__(self, n):
        self.f = [0] * (n + 2)
        self.n = n + 2

    def add(self, i, v=1):
        i += 1
        while i < self.n:
            self.f[i] += v
            i += i & -i

    def sum(self, i):
        i += 1
        r = 0
        while i > 0:
            r += self.f[i]
            i -= i & -i
        return r


class Solution:
    def kBigIndices(self, nums: List[int], k: int) -> int:
        # compress
        vals = sorted(set(nums))
        mp = {v: i for i, v in enumerate(vals)}
        n = len(nums)
        left = [0] * n
        bit = BIT(len(vals))
        for i, x in enumerate(nums):
            left[i] = bit.sum(mp[x] - 1)
            bit.add(mp[x])
        bit = BIT(len(vals))
        ans = 0
        for i in range(n - 1, -1, -1):
            right = bit.sum(mp[nums[i]] - 1)
            if left[i] >= k and right >= k:
                ans += 1
            bit.add(mp[nums[i]])
        return ans


def _test():
    s = Solution()
    assert s.kBigIndices([2, 3, 6, 5, 2, 3], 2) == 2
    print("LC 2519 OK")


if __name__ == "__main__":
    _test()
