"""
LC 1359. Count All Valid Pickup and Delivery Options — Hard
https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/
Best: Combinatorial DP — O(n)
"""


class Solution:
    def countOrders(self, n: int) -> int:
        MOD = 10**9 + 7
        ans = 1
        for i in range(2, n + 1):
            # insert PiDi among 2(i-1)+1 slots for Pi, then Di after Pi
            ans = ans * (2 * i - 1) * i % MOD
        return ans


def _test():
    s = Solution()
    assert s.countOrders(1) == 1
    assert s.countOrders(2) == 6
    assert s.countOrders(3) == 90
    print("LC 1359 OK")


if __name__ == "__main__":
    _test()
