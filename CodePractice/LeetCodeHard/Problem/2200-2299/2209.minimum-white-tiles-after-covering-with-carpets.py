"""
LC 2209. Minimum White Tiles After Covering With Carpets — Hard
https://leetcode.com/problems/minimum-white-tiles-after-covering-with-carpets/
Best: DP cover white tiles with carpets — O(n*numCarpets)
"""


class Solution:
    def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
        n = len(floor)
        # pref white
        white = [0] * (n + 1)
        for i, c in enumerate(floor):
            white[i + 1] = white[i] + (c == "1")
        # dp[i][j] = min remaining white using j carpets on floor[i:]
        INF = 10**9
        dp = [[0] * (numCarpets + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(numCarpets + 1):
                # skip
                best = dp[i + 1][j] + (floor[i] == "1")
                if j > 0:
                    ni = min(n, i + carpetLen)
                    best = min(best, dp[ni][j - 1])
                dp[i][j] = best
        return dp[0][numCarpets]


def _test():
    s = Solution()
    assert s.minimumWhiteTiles("10110101", 2, 2) == 2
    assert s.minimumWhiteTiles("11111", 2, 3) == 0
    print("LC 2209 OK")


if __name__ == "__main__":
    _test()

