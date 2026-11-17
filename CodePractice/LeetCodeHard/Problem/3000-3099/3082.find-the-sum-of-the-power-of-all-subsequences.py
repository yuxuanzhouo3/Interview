"""
LC 3082. Find the Sum of the Power of All Subsequences — Hard
https://leetcode.com/problems/find-the-sum-of-the-power-of-all-subsequences/
Best: Knapsack DP count subsets with sum k, then * 2^{n-|S|} — O(n·k)
"""

from typing import List

MOD = 10**9 + 7


class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[s] = number of subsequences with sum s (as subsets of chosen elements)
        dp = [0] * (k + 1)
        dp[0] = 1
        for x in nums:
            for s in range(k, x - 1, -1):
                dp[s] = (dp[s] + dp[s - x]) % MOD
        # Each subsequence S with sum(S)=k contributes 2^{n-|S|}? 
        # Power = number of subsequences of S that sum to k.
        # Total = sum over all T ⊆ nums of (number of S ⊆ T with sum(S)=k)
        # = sum_{S: sum=k} 2^{n-|S|}
        # Need to track size: use dp2
        # Recompute with size
        # dp[j][s] too heavy; instead online:
        # When adding element, we need contribution weighted by 2^{remaining}
        # Standard: ans = sum_S 2^{n-|S|} for sum(S)=k
        # Process left-to-right maintaining count of ways to make sum with used count
        ways = [0] * (k + 1)  # ways[s] = Σ 2^{-|S|} style → multiply by 2^n at end
        # Let f[s] = sum over S sum=s of 2^{n-|S|} / 2^n = sum 2^{-|S|}
        # When considering elements one by one:
        f = [0] * (k + 1)
        f[0] = 1  # empty, weight 1 (= 2^0 / something). Track: f[s] = Σ_{sum S=s} 2^{processed - |S|} after processed items? 
        # Cleaner approach from editorial:
        # dp[j] = number of ways to get sum j using some subset; then for each subset S with sum k,
        # remaining elements each free → 2^{n-|S|}.
        # So maintain dp[j][c] count of subsets with sum j size c — n*k*n too big? n<=100,k<=100 → ok O(n^2 k)
        ndp = [[0] * (n + 1) for _ in range(k + 1)]
        ndp[0][0] = 1
        for x in nums:
            for s in range(k, x - 1, -1):
                for c in range(n, 0, -1):
                    ndp[s][c] = (ndp[s][c] + ndp[s - x][c - 1]) % MOD
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = pow2[i - 1] * 2 % MOD
        ans = 0
        for c in range(n + 1):
            ans = (ans + ndp[k][c] * pow2[n - c]) % MOD
        return ans


def _test():
    s = Solution()
    assert s.sumOfPower([1, 2, 3], 3) == 6
    assert s.sumOfPower([2, 3, 3], 5) == 4
    assert s.sumOfPower([1, 2, 3], 7) == 0
    print("LC 3082 OK")


if __name__ == "__main__":
    _test()
