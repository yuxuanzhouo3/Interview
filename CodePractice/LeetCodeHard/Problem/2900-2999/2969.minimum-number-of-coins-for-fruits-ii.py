"""
LC 2969. Minimum Number of Coins for Fruits II — Hard
https://leetcode.com/problems/minimum-number-of-coins-for-fruits-ii/
Best: Monotonic deque DP — O(n)
"""

from collections import deque
from typing import List


class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] = min cost to acquire fruits i..n-1 (0-index), must buy i
        # After buying i (1-index i+1), get next i fruits free: i+1 .. i+1+i = 2i+1 in 1-index
        # 0-index: can cover up to min(n-1, i + (i+1)) = min(n-1, 2*i+1)
        dp = [0] * (n + 1)
        dq = deque()  # increasing dp indices for future
        for i in range(n - 1, -1, -1):
            # remove indices j where we cannot reach from buying i: j > 2*i+1
            while dq and dq[0] > 2 * i + 1:
                dq.popleft()
            # next fruit to pay after free range starts at 2*i+2
            nxt = 2 * i + 2
            if nxt >= n:
                dp[i] = prices[i]
            else:
                while dq and dq[0] < nxt:
                    # need min dp[j] for j in [nxt, 2*i+1]? Actually free gets fruits i+1..2i+1,
                    # so first unpaid is 2i+2; dp[i]=prices[i]+min(dp[j] for j>=2i+2) but also
                    # can jump within free? Standard: dp[i]=prices[i]+min(dp[i+1..min(n,2i+2)])
                    pass
                # Correct formulation from 2944:
                # dp[i] = prices[i] + min(dp[j] for j = i+1 .. min(n, 2*(i+1)))
                # where dp[n]=0
                pass
            # rebuild properly below
        # Rewrite cleanly
        dp = [0] * (n + 1)
        dq = deque([n])  # stores indices with increasing dp
        dp[n] = 0
        for i in range(n - 1, -1, -1):
            # window of j in [i+1, min(n, 2*(i+1))]
            r = min(n, 2 * (i + 1))
            while dq and dq[0] > r:
                dq.popleft()
            dp[i] = prices[i] + dp[dq[0]]
            while dq and dp[i] <= dp[dq[-1]]:
                dq.pop()
            dq.append(i)
        return dp[0]


def _test():
    s = Solution()
    assert s.minimumCoins([3, 1, 2]) == 4
    assert s.minimumCoins([1, 10, 1, 1]) == 2
    print("LC 2969 OK")


if __name__ == "__main__":
    _test()

