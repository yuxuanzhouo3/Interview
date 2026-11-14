"""
LC 960. Delete Columns to Make Sorted III — Hard
https://leetcode.com/problems/delete-columns-to-make-sorted-iii/
Best: LIS-like column DP — O(m^2 * n)
"""

from typing import List


class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        n, m = len(strs), len(strs[0])
        # dp[j] = longest kept increasing column subsequence ending at col j
        dp = [1] * m
        for j in range(m):
            for i in range(j):
                if all(strs[r][i] <= strs[r][j] for r in range(n)):
                    dp[j] = max(dp[j], dp[i] + 1)
        return m - max(dp)


def _test():
    s = Solution()
    assert s.minDeletionSize(["babca", "bbazb"]) == 3
    assert s.minDeletionSize(["edcba"]) == 4
    assert s.minDeletionSize(["ghi", "def", "abc"]) == 0
    print("LC 960 OK")


if __name__ == "__main__":
    _test()
