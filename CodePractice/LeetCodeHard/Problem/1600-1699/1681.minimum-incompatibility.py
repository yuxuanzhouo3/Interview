"""
LC 1681. Minimum Incompatibility — Hard
https://leetcode.com/problems/minimum-incompatibility/
Best: Subset DP on valid groups — O(3^n)
"""

from typing import List


class Solution:
    def minimumIncompatibility(self, nums: List[int], k: int) -> int:
        n = len(nums)
        group = n // k
        nums.sort()
        INF = 10**9
        N = 1 << n
        # valid[mask] = incompatibility if mask is valid group else INF
        valid = [INF] * N
        for mask in range(N):
            if bin(mask).count("1") != group:
                continue
            vals = [nums[i] for i in range(n) if mask & (1 << i)]
            if len(set(vals)) != group:
                continue
            valid[mask] = max(vals) - min(vals)

        dp = [INF] * N
        dp[0] = 0
        for mask in range(N):
            if dp[mask] == INF:
                continue
            # pick next group from remaining: take lowest unset bit start
            remain = ((1 << n) - 1) ^ mask
            # enumerate submasks of remain with size group
            sub = remain
            while sub:
                if valid[sub] < INF:
                    dp[mask | sub] = min(dp[mask | sub], dp[mask] + valid[sub])
                sub = (sub - 1) & remain

        return dp[N - 1] if dp[N - 1] < INF else -1


def _test():
    s = Solution()
    assert s.minimumIncompatibility([1, 2, 1, 4], 2) == 4
    assert s.minimumIncompatibility([6, 3, 8, 1, 3, 1, 2, 2], 4) == 6
    assert s.minimumIncompatibility([5, 3, 3, 6, 3, 3], 3) == -1
    print("LC 1681 OK")


if __name__ == "__main__":
    _test()
