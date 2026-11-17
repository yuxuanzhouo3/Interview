"""
LC 2584. Split the Array to Make Coprime Products — Hard
https://leetcode.com/problems/split-the-array-to-make-coprime-products/
Best: Prime factor rightmost reach — O(n log A)
"""

from typing import List
from math import gcd


class Solution:
    def findValidSplit(self, nums: List[int]) -> int:
        n = len(nums)
        # last index of each prime factor
        last = {}
        def factors(x):
            f = []
            d = 2
            while d * d <= x:
                if x % d == 0:
                    f.append(d)
                    while x % d == 0:
                        x //= d
                d += 1
            if x > 1:
                f.append(x)
            return f
        facs = [factors(x) for x in nums]
        for i, fs in enumerate(facs):
            for p in fs:
                last[p] = i
        reach = 0
        for i in range(n - 1):
            for p in facs[i]:
                reach = max(reach, last[p])
            if reach == i:
                return i
        return -1


def _test():
    s = Solution()
    assert s.findValidSplit([4, 7, 8, 15, 3, 5]) == 2
    assert s.findValidSplit([4, 7, 15, 8, 3, 5]) == -1
    print("LC 2584 OK")


if __name__ == "__main__":
    _test()

