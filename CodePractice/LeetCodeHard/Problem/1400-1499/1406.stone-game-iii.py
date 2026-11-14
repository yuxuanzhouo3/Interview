"""
LC 1406. Stone Game III — Hard
https://leetcode.com/problems/stone-game-iii/
Best: Suffix DP — O(n)
"""

from typing import List


class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        n = len(stoneValue)
        dp = [0] * (n + 1)  # best score diff Alice-Bob from i
        for i in range(n - 1, -1, -1):
            best = float("-inf")
            take = 0
            for k in range(3):
                if i + k >= n:
                    break
                take += stoneValue[i + k]
                best = max(best, take - dp[i + k + 1])
            dp[i] = best
        if dp[0] > 0:
            return "Alice"
        if dp[0] < 0:
            return "Bob"
        return "Tie"


def _test():
    s = Solution()
    assert s.stoneGameIII([1, 2, 3, 7]) == "Bob"
    assert s.stoneGameIII([1, 2, 3, -9]) == "Alice"
    assert s.stoneGameIII([1, 2, 3, 6]) == "Tie"
    print("LC 1406 OK")


if __name__ == "__main__":
    _test()
