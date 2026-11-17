"""
LC 3139. Minimum Cost to Equalize Array — Hard
https://leetcode.com/problems/minimum-cost-to-equalize-array/
Best: Enumerate target + pair ops — O(n + Δ)
"""

from typing import List


class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        M = max(nums)
        m = min(nums)
        s = sum(nums)
        total = M * n - s
        if cost2 >= 2 * cost1 or n <= 2:
            return total * cost1 % MOD
        ans = total * cost1
        for T in range(M, 2 * M - m + 1):
            S = T * n - s
            mxd = T - m
            pairs = min(S // 2, S - mxd)
            ans = min(ans, pairs * cost2 + (S - 2 * pairs) * cost1)
        return ans % MOD


def _test():
    s = Solution()
    assert s.minCostToEqualizeArray([1, 14, 14, 15], 2, 1) == 20
    assert s.minCostToEqualizeArray([4, 1], 5, 2) == 15
    assert s.minCostToEqualizeArray([1, 1, 1, 1, 1], 1, 1) == 0
    print("LC 3139 OK")


if __name__ == "__main__":
    _test()
