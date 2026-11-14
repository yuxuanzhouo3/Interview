"""
LC 123. Best Time to Buy and Sell Stock III — Hard
Best: State machine DP — at most 2 transactions
"""

from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy1 = buy2 = float("inf")
        sell1 = sell2 = 0
        for p in prices:
            buy1 = min(buy1, p)
            sell1 = max(sell1, p - buy1)
            buy2 = min(buy2, p - sell1)
            sell2 = max(sell2, p - buy2)
        return sell2


def _test():
    s = Solution()
    assert s.maxProfit([3, 3, 5, 0, 0, 3, 1, 4]) == 6
    assert s.maxProfit([1, 2, 3, 4, 5]) == 4
    print("LC 123 OK")


if __name__ == "__main__":
    _test()
