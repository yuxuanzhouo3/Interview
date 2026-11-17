"""
LC 1510. Stone Game IV — Hard
https://leetcode.com/problems/stone-game-iv/
Best: DP win/lose — O(n √n)
"""


class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        dp = [False] * (n + 1)
        for i in range(1, n + 1):
            x = 1
            while x * x <= i:
                if not dp[i - x * x]:
                    dp[i] = True
                    break
                x += 1
        return dp[n]


def _test():
    s = Solution()
    assert s.winnerSquareGame(1) is True
    assert s.winnerSquareGame(2) is False
    assert s.winnerSquareGame(4) is True
    assert s.winnerSquareGame(7) is False
    print("LC 1510 OK")


if __name__ == "__main__":
    _test()
