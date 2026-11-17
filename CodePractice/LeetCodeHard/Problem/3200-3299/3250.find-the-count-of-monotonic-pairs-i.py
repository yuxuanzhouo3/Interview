"""
LC 3250. Find the Count of Monotonic Pairs I — Hard
https://leetcode.com/problems/find-the-count-of-monotonic-pairs-i/
Best: DP + prefix sums — O(n * max(nums))
"""

from typing import List


class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        mx = max(nums)
        dp = [1] * (nums[0] + 1) + [0] * (mx - nums[0])
        for i in range(1, len(nums)):
            pref = [0]
            for x in dp:
                pref.append((pref[-1] + x) % MOD)
            ndp = [0] * (mx + 1)
            diff = max(0, nums[i] - nums[i - 1])
            for v in range(nums[i] + 1):
                up = v - diff
                if up >= 0:
                    ndp[v] = pref[min(up, nums[i - 1]) + 1] % MOD
            dp = ndp
        return sum(dp[: nums[-1] + 1]) % MOD


def _test():
    s = Solution()
    assert s.countOfPairs([2, 3, 2]) == 4
    assert s.countOfPairs([5, 5, 5, 5]) == 126
    print("LC 3250 OK")


if __name__ == "__main__":
    _test()
