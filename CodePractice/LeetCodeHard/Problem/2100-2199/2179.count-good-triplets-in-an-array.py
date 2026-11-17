"""
LC 2179. Count Good Triplets in an Array — Hard
https://leetcode.com/problems/count-good-triplets-in-an-array/
Best: BIT count left smaller & right larger — O(n log n)
"""


from typing import List


class BIT:
    def __init__(self, n):
        self.n = n
        self.f = [0] * (n + 1)

    def add(self, i, v=1):
        while i <= self.n:
            self.f[i] += v
            i += i & -i

    def sum(self, i):
        s = 0
        while i > 0:
            s += self.f[i]
            i -= i & -i
        return s


class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        pos2 = [0] * n
        for i, x in enumerate(nums2):
            pos2[x] = i
        # map nums1 values to positions in nums2
        arr = [pos2[x] for x in nums1]
        bit = BIT(n)
        left = [0] * n
        for i, v in enumerate(arr):
            left[i] = bit.sum(v)  # how many previous with smaller pos (0-index -> v is 0..n-1)
            bit.add(v + 1)
        bit = BIT(n)
        right = [0] * n
        for i in range(n - 1, -1, -1):
            v = arr[i]
            right[i] = bit.sum(n) - bit.sum(v + 1)  # larger positions after
            bit.add(v + 1)
        return sum(left[i] * right[i] for i in range(n))


def _test():
    s = Solution()
    assert s.goodTriplets([2, 0, 1, 3], [0, 1, 2, 3]) == 1
    assert s.goodTriplets([4, 0, 1, 3, 2], [4, 1, 0, 2, 3]) == 4
    print("LC 2179 OK")


if __name__ == "__main__":
    _test()

