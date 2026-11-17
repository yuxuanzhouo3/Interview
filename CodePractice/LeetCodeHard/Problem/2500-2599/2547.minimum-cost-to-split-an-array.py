"""
LC 2547. Minimum Cost to Split an Array — Hard
https://leetcode.com/problems/minimum-cost-to-split-an-array/
Best: DP + trimmed unique count — O(n^2)
"""

from typing import List


class Solution:
    def minCost(self, nums: List[int], k: int) -> int:
        n = len(nums)
        INF = 10**18
        dp = [0] + [INF] * n
        for i in range(n):
            freq = {}
            trimmed = 0
            best = INF
            for j in range(i, -1, -1):
                v = nums[j]
                c = freq.get(v, 0)
                if c == 0:
                    freq[v] = 1
                elif c == 1:
                    freq[v] = 2
                    trimmed += 2
                else:
                    freq[v] = c + 1
                    trimmed += 1
                best = min(best, dp[j] + k + trimmed)
            dp[i + 1] = best
        return dp[n]


def _test():
    s = Solution()
    assert s.minCost([1, 2, 1, 2, 1, 3, 3], 2) == 8
    assert s.minCost([1, 2, 1, 2, 1], 2) == 6
    assert s.minCost([1, 2, 1, 2, 1], 5) == 10
    print("LC 2547 OK")


if __name__ == "__main__":
    _test()

