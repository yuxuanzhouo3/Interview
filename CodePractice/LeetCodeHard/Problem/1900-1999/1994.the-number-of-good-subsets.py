"""
LC 1994. The Number of Good Subsets — Hard
https://leetcode.com/problems/the-number-of-good-subsets/
Best: bitmask DP on 10 primes — O(2^10 * 30)
"""

from typing import List
from collections import Counter


class Solution:
    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        cnt = Counter(nums)
        # mask for numbers 1..30 that are square-free
        num_mask = {}
        for x in range(2, 31):
            m = 0
            ok = True
            y = x
            for i, p in enumerate(primes):
                if y % p == 0:
                    c = 0
                    while y % p == 0:
                        y //= p
                        c += 1
                    if c >= 2:
                        ok = False
                        break
                    m |= 1 << i
            if ok and y == 1:
                num_mask[x] = m
        dp = [0] * (1 << 10)
        dp[0] = 1
        for x, mask in num_mask.items():
            if cnt[x] == 0:
                continue
            for s in range((1 << 10) - 1, -1, -1):
                if s & mask:
                    continue
                dp[s | mask] = (dp[s | mask] + dp[s] * cnt[x]) % MOD
        ans = sum(dp[1:]) % MOD  # exclude empty
        # multiply by 2^{cnt[1]}
        ones = pow(2, cnt[1], MOD)
        ans = ans * ones % MOD
        return ans


def _test():
    s = Solution()
    assert s.numberOfGoodSubsets([1, 2, 3, 4]) == 6
    assert s.numberOfGoodSubsets([4, 4, 4, 5, 6]) == 3
    print("LC 1994 OK")


if __name__ == "__main__":
    _test()
