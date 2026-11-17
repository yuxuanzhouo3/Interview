"""
LC 2518. Number of Great Partitions — Hard
https://leetcode.com/problems/number-of-great-partitions/
Best: Subset-sum DP subtract invalid — O(nk)
"""

from typing import List


class Solution:
    def countPartitions(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        total = sum(nums)
        if total < 2 * k:
            return 0
        ans = pow(2, len(nums), MOD)
        dp = [1] + [0] * (k - 1)
        for num in nums:
            if num >= k:
                continue
            for s in range(k - 1, num - 1, -1):
                dp[s] = (dp[s] + dp[s - num]) % MOD
        bad = sum(dp) % MOD
        return (ans - 2 * bad) % MOD


def _test():
    s = Solution()
    assert s.countPartitions([1, 2, 3, 4], 4) == 6
    assert s.countPartitions([3, 3, 3], 4) == 0
    assert s.countPartitions([6, 6], 2) == 2
    print("LC 2518 OK")


if __name__ == "__main__":
    _test()

