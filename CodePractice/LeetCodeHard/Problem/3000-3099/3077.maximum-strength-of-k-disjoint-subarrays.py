"""
LC 3077. Maximum Strength of K Disjoint Subarrays — Hard
https://leetcode.com/problems/maximum-strength-of-k-disjoint-subarrays/
Best: DP with k segments alternating sign — O(n k)
"""

from typing import List


class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # strength = sum_j ((-1)^(j+1) * sum(subarray_j) * (k-j+1))
        # = sum_j coeff[j] * sum(seg_j), coeff[j] = (-1)^(j+1)*(k-j+1)
        # DP: dp[i][j] = max strength using first i elems and j segments
        # Two states: ending with open j-th segment or closed
        NEG = -(10**18)
        # dp_end[j] = best with j segments, last segment ends at current
        # dp_best[j] = best with j segments anywhere in prefix
        dp_best = [NEG] * (k + 1)
        dp_end = [NEG] * (k + 1)
        dp_best[0] = 0
        for i, x in enumerate(nums):
            ndp_best = dp_best[:]
            ndp_end = [NEG] * (k + 1)
            for j in range(1, k + 1):
                coeff = (k - j + 1) * (1 if j % 2 == 1 else -1)
                # extend previous end
                if dp_end[j] > NEG:
                    ndp_end[j] = max(ndp_end[j], dp_end[j] + coeff * x)
                # start new segment j at i
                if dp_best[j - 1] > NEG:
                    ndp_end[j] = max(ndp_end[j], dp_best[j - 1] + coeff * x)
                if ndp_end[j] > NEG:
                    ndp_best[j] = max(ndp_best[j], ndp_end[j])
            dp_best, dp_end = ndp_best, ndp_end
        return dp_best[k]


def _test():
    s = Solution()
    assert s.maximumStrength([1, 2, 3, -1, 2], 3) == 22
    assert s.maximumStrength([12, -2, -2, -2, -2], 5) == 64
    assert s.maximumStrength([-1, -2, -3], 1) == -1
    print("LC 3077 OK")


if __name__ == "__main__":
    _test()
