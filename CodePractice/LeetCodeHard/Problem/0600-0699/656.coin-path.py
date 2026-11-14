"""
LC 656. Coin Path — Hard
https://leetcode.com/problems/coin-path/
Best: DP cheapest path with ≤B jumps — O(nB)
"""

from typing import List


class Solution:
    def cheapestJump(self, coins: List[int], maxJump: int) -> List[int]:
        n = len(coins)
        if coins[-1] < 0:
            return []
        # cost[i], next_idx[i]
        INF = 10**18
        cost = [INF] * n
        nxt = [-1] * n
        cost[n - 1] = coins[n - 1]
        for i in range(n - 2, -1, -1):
            if coins[i] < 0:
                continue
            for j in range(i + 1, min(n, i + maxJump + 1)):
                if cost[j] == INF:
                    continue
                cand = coins[i] + cost[j]
                if cand < cost[i] or (cand == cost[i] and (nxt[i] == -1 or j < nxt[i])):
                    cost[i] = cand
                    nxt[i] = j
        if cost[0] == INF:
            return []
        path = []
        i = 0
        while i != -1:
            path.append(i + 1)  # 1-indexed
            if i == n - 1:
                break
            i = nxt[i]
        return path


def _test():
    s = Solution()
    assert s.cheapestJump([1, 2, 4, -1, 2], 2) == [1, 3, 5]
    assert s.cheapestJump([1, 2, 4, -1, 2], 1) == []
    print("LC 656 OK")


if __name__ == "__main__":
    _test()
