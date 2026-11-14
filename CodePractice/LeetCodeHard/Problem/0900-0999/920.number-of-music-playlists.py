"""
LC 920. Number of Music Playlists — Hard
https://leetcode.com/problems/number-of-music-playlists/
Best: DP — O(n * goal)
"""

class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        MOD = 10**9 + 7
        # dp[i][j] = playlists of length i using exactly j distinct songs
        dp = [[0] * (n + 1) for _ in range(goal + 1)]
        dp[0][0] = 1
        for i in range(1, goal + 1):
            for j in range(1, min(i, n) + 1):
                # play a new song
                dp[i][j] = dp[i - 1][j - 1] * (n - j + 1) % MOD
                # replay an old song (must keep >=k gap: j-k choices)
                if j > k:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j] * (j - k)) % MOD
        return dp[goal][n]


def _test():
    s = Solution()
    assert s.numMusicPlaylists(3, 3, 1) == 6
    assert s.numMusicPlaylists(2, 3, 0) == 6
    assert s.numMusicPlaylists(2, 3, 1) == 2
    print("LC 920 OK")


if __name__ == "__main__":
    _test()
