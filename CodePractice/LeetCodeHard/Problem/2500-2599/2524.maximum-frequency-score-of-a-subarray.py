"""
LC 2524. Maximum Frequency Score of a Subarray — Hard
Best: Sliding window + pow mod — O(n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        freq = defaultdict(int)
        score = 0
        ans = 0

        def modpow(a, b):
            return pow(a, b, MOD)

        for i, x in enumerate(nums):
            # add x
            if freq[x]:
                score = (score - modpow(x, freq[x])) % MOD
            freq[x] += 1
            score = (score + modpow(x, freq[x])) % MOD
            if i >= k:
                y = nums[i - k]
                score = (score - modpow(y, freq[y])) % MOD
                freq[y] -= 1
                if freq[y]:
                    score = (score + modpow(y, freq[y])) % MOD
            if i >= k - 1:
                ans = max(ans, score)
        return ans


def _test():
    s = Solution()
    assert s.maxFrequencyScore([1, 1, 1], 3) == 1  # 1^3=1? wait 1^3=1
    print("LC 2524", s.maxFrequencyScore([1, 1, 1], 2), "OK")


if __name__ == "__main__":
    _test()
