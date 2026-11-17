"""
LC 1223. Dice Roll Simulation — Hard
https://leetcode.com/problems/dice-roll-simulation/
Best: DP — O(n · 6 · max(rollMax))
"""

from typing import List


class Solution:
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        MOD = 10**9 + 7
        # dp[i][j][k] = ways for i rolls, last face j, consecutive k times
        # compress: dp[face][consec]
        dp = [[0] * 16 for _ in range(6)]
        for j in range(6):
            dp[j][1] = 1
        for _ in range(2, n + 1):
            ndp = [[0] * 16 for _ in range(6)]
            for j in range(6):
                for k in range(1, rollMax[j] + 1):
                    if dp[j][k] == 0:
                        continue
                    for nj in range(6):
                        if nj == j:
                            if k + 1 <= rollMax[j]:
                                ndp[nj][k + 1] = (ndp[nj][k + 1] + dp[j][k]) % MOD
                        else:
                            ndp[nj][1] = (ndp[nj][1] + dp[j][k]) % MOD
            dp = ndp
        return sum(dp[j][k] for j in range(6) for k in range(16)) % MOD


def _test():
    s = Solution()
    assert s.dieSimulator(2, [1, 1, 2, 2, 2, 3]) == 34
    assert s.dieSimulator(2, [1, 1, 1, 1, 1, 1]) == 30
    assert s.dieSimulator(3, [1, 1, 1, 2, 2, 3]) == 181
    print("LC 1223 OK")


if __name__ == "__main__":
    _test()
