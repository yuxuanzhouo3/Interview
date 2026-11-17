"""
LC 2412. Minimum Money Required Before Transactions — Hard
https://leetcode.com/problems/minimum-money-required-before-transactions/
Best: Sum losses + max cash-on-hand peak — O(n)
"""

from typing import List


class Solution:
    def minimumMoney(self, transactions: List[List[int]]) -> int:
        total = 0
        mx = 0
        for cost, cashback in transactions:
            if cost > cashback:
                total += cost - cashback
                mx = max(mx, cashback)
            else:
                mx = max(mx, cost)
        return total + mx


def _test():
    s = Solution()
    assert s.minimumMoney([[2, 1], [5, 0], [4, 2]]) == 10
    assert s.minimumMoney([[3, 0], [0, 3]]) == 3
    print("LC 2412 OK")


if __name__ == "__main__":
    _test()
