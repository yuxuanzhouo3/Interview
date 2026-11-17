"""
LC 2921. Maximum Profitable Triplets With Increasing Prices II — Hard
Best: Fenwick max profit by price — O(n log n)
"""

from typing import List


class BIT:
    def __init__(self, n):
        self.f = [0] * (n + 2)
        self.n = n + 2

    def update(self, i, v):
        i += 1
        while i < self.n:
            self.f[i] = max(self.f[i], v)
            i += i & -i

    def query(self, i):
        i += 1
        r = 0
        while i > 0:
            r = max(r, self.f[i])
            i -= i & -i
        return r


class Solution:
    def maxProfit(self, prices: List[int], profits: List[int]) -> int:
        vals = sorted(set(prices))
        mp = {v: i for i, v in enumerate(vals)}
        bit = BIT(len(vals))
        best2 = [0] * len(prices)  # max p_i+p_j for j<i price_j<price_i
        ans = -1
        for i, (p, pr) in enumerate(zip(prices, profits)):
            left = bit.query(mp[p] - 1)
            if left:
                best2[i] = left + pr
            bit.update(mp[p], pr)
        bit = BIT(len(vals))
        for i in range(len(prices) - 1, -1, -1):
            p, pr = prices[i], profits[i]
            # need price > p with profit; query from high side - use reversed index
            # rebuild with reverse prices
            pass
        # second pass forward for triplets using best2
        bit2 = BIT(len(vals))
        for i, (p, pr) in enumerate(zip(prices, profits)):
            if best2[i]:
                # find max profit with price > p? triplet i left two + right
                pass
        # simpler O(n^2) for moderate n - constraints n<=1e5 need BIT
        # left max profit strictly less price
        left = [0] * len(prices)
        bit = BIT(len(vals))
        for i, (p, pr) in enumerate(zip(prices, profits)):
            left[i] = bit.query(mp[p] - 1)
            bit.update(mp[p], pr)
        right = [0] * len(prices)
        # BIT on reversed rank
        bit = BIT(len(vals))
        for i in range(len(prices) - 1, -1, -1):
            p, pr = prices[i], profits[i]
            # query prices > p => rank > mp[p]
            # store at reversed index
            ri = len(vals) - 1 - mp[p]
            right[i] = bit.query(ri - 1)
            bit.update(ri, pr)
        for i in range(len(prices)):
            if left[i] and right[i]:
                ans = max(ans, left[i] + profits[i] + right[i])
        return ans


def _test():
    s = Solution()
    assert s.maxProfit([10, 2, 3, 4], [100, 2, 7, 10]) == 19
    print("LC 2921 OK")


if __name__ == "__main__":
    _test()
