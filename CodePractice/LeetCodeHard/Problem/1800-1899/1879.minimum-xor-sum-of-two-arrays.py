"""
LC 1879. Minimum XOR Sum of Two Arrays — Hard
https://leetcode.com/problems/minimum-xor-sum-of-two-arrays/
Best: Bitmask DP assign nums2 — O(n^2 2^n)
"""


from typing import List


class Solution:
    def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        INF = 10**18
        dp = [INF] * (1 << n)
        dp[0] = 0
        for mask in range(1 << n):
            i = bin(mask).count("1")
            if i >= n:
                continue
            for j in range(n):
                if mask & (1 << j) == 0:
                    nm = mask | (1 << j)
                    dp[nm] = min(dp[nm], dp[mask] + (nums1[i] ^ nums2[j]))
        return dp[(1 << n) - 1]


def _test():
    s = Solution()
    assert s.minimumXORSum([1, 2], [2, 3]) == 2
    assert s.minimumXORSum([1, 0, 3], [5, 3, 4]) == 8
    print("LC 1879 OK")


if __name__ == "__main__":
    _test()

