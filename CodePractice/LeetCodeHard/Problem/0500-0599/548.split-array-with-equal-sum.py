"""
LC 548. Split Array with Equal Sum — Hard
https://leetcode.com/problems/split-array-with-equal-sum/
Tags: Array, Hash Table, Prefix Sum
Best: 前缀和 + 枚举中切，集合存左等分
"""

from typing import List


class Solution:
    def splitArray(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 7:
            return False
        pref = [0] * (n + 1)
        for i, x in enumerate(nums):
            pref[i + 1] = pref[i] + x

        def s(i, j):
            return pref[j + 1] - pref[i]

        for j in range(3, n - 3):
            seen = set()
            for i in range(1, j - 1):
                if s(0, i - 1) == s(i + 1, j - 1):
                    seen.add(s(0, i - 1))
            for k in range(j + 2, n - 1):
                right = s(j + 1, k - 1)
                if right == s(k + 1, n - 1) and right in seen:
                    return True
        return False


def _test():
    sol = Solution()
    assert sol.splitArray([1, 2, 1, 2, 1, 2, 1]) is True
    assert sol.splitArray([1, 2, 1, 2, 1, 2, 1, 2]) is False
    print("LC 548 OK")


if __name__ == "__main__":
    _test()

