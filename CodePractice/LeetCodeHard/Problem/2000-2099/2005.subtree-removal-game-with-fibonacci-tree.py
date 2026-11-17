"""
LC 2005. Subtree Removal Game with Fibonacci Tree — Hard
https://leetcode.com/problems/subtree-removal-game-with-fibonacci-tree/
Best: SG / 观察：Alice 当且仅当 n % 2 == 1 — O(1)
"""


class Solution:
    def findGameWinner(self, n: int) -> bool:
        # Fibonacci tree F_n: left F_{n-2}, right F_{n-1}
        # Grundy analysis shows first player wins iff n is odd
        return n % 2 == 1


def _test():
    s = Solution()
    assert s.findGameWinner(3) is True
    assert s.findGameWinner(1) is True
    assert s.findGameWinner(2) is False
    assert s.findGameWinner(4) is False
    print("LC 2005 OK")


if __name__ == "__main__":
    _test()
