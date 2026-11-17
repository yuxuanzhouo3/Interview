"""
LC 891. Sum of Subsequence Widths — Hard
Best: Sort + each as max/min contribution — O(n log n)
"""

from typing import List

MOD = 10**9 + 7


class Solution:
    def sumSubseqWidths(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        pow2 = [1] * n
        for i in range(1, n):
            pow2[i] = pow2[i - 1] * 2 % MOD
        ans = 0
        for i, x in enumerate(nums):
            ans = (ans + x * pow2[i] - x * pow2[n - 1 - i]) % MOD
        return ans


def _test():
    s = Solution()
    assert s.sumSubseqWidths([2, 1, 3]) == 6
    print("LC 891 OK")


if __name__ == "__main__":
    _test()
