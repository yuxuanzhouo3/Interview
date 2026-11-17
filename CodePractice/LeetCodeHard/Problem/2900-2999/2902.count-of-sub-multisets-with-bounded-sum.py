"""
LC 2902. Count of Sub-Multisets With Bounded Sum — Hard
https://leetcode.com/problems/count-of-sub-multisets-with-bounded-sum/
Best: Frequency + binary/grouped knapsack DP — O(r * distinct)
"""

from typing import List
from collections import Counter


class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        cnt = Counter(nums)
        zeros = cnt.pop(0, 0)
        dp = [0] * (r + 1)
        dp[0] = 1
        for val, c in cnt.items():
            ndp = dp[:]
            for rem in range(val):
                window = 0
                # sliding: for s = rem, rem+val, ...
                # use prefix of dp at rem + k*val
                for k in range((r - rem) // val + 1):
                    s = rem + k * val
                    window = (window + dp[s]) % MOD
                    if k > c:
                        window = (window - dp[rem + (k - c - 1) * val]) % MOD
                    if k >= 1:
                        # ndp[s] already has dp[s] (take 0 of this val); add ways using 1..min(c,k) copies
                        # actually rebuild: ways to make s using this number
                        pass
            # standard multiple knapsack with sliding window
            ndp = dp[:]
            for rem in range(val):
                qsum = 0
                # process in order of increasing count
                for k in range((r - rem) // val + 1):
                    s = rem + k * val
                    qsum = (qsum + dp[s]) % MOD
                    if k - c - 1 >= 0:
                        qsum = (qsum - dp[rem + (k - c - 1) * val]) % MOD
                    ndp[s] = qsum
            dp = ndp
        ans = sum(dp[l:r + 1]) % MOD
        # zeros: each subset can include 0..zeros zeros → multiply (zeros+1)
        ans = ans * (zeros + 1) % MOD
        return ans


def _test():
    s = Solution()
    assert s.countSubMultisets([1, 2, 2, 3], 6, 6) == 1
    assert s.countSubMultisets([2, 1, 4, 2, 7], 1, 5) == 7
    assert s.countSubMultisets([1, 2, 1, 3, 5, 2], 3, 5) == 9
    print("LC 2902 OK")


if __name__ == "__main__":
    _test()
