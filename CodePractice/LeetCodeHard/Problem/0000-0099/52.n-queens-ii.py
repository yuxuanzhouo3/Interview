"""
LC 52. N-Queens II — Hard
Best: Bitmask DFS count — O(n!)
"""

class Solution:
    def totalNQueens(self, n: int) -> int:
        self.ans = 0

        def dfs(row, cols, d1, d2):
            if row == n:
                self.ans += 1
                return
            available = ~(cols | d1 | d2) & ((1 << n) - 1)
            while available:
                bit = available & -available
                available -= bit
                dfs(row + 1, cols | bit, (d1 | bit) << 1, (d2 | bit) >> 1)

        dfs(0, 0, 0, 0)
        return self.ans


def _test():
    s = Solution()
    assert s.totalNQueens(4) == 2
    assert s.totalNQueens(1) == 1
    print("LC 52 OK")


if __name__ == "__main__":
    _test()
