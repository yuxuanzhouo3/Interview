"""
LC 1872. Stone Game VIII — Hard
https://leetcode.com/problems/stone-game-viii/
Best: prefix + DP best = max(prefix[i]-best_prev) — O(n)
"""

from typing import List


class Solution:
    def stoneGameVIII(self, stones: List[int]) -> int:
        n = len(stones)
        pref = [0] * n
        pref[0] = stones[0]
        for i in range(1, n):
            pref[i] = pref[i - 1] + stones[i]
        # dp: Alice's best score difference when stones are reduced to prefix ending at i..n-1
        # best = max over i of pref[i] - best_after_opponent
        # From the end: taking all remaining -> score pref[n-1]
        best = pref[n - 1]
        for i in range(n - 2, 0, -1):
            # can stop at prefix i (remove first i+1 stones as one stone = pref[i])
            best = max(best, pref[i] - best)
        return best


def _test():
    s = Solution()
    assert s.stoneGameVIII([-1, 2, -3, 4, -5]) == 5
    assert s.stoneGameVIII([7, -6, 5, 10, 5, -2, -6]) == 13
    assert s.stoneGameVIII([-10, -12]) == -22
    print("LC 1872 OK")


if __name__ == "__main__":
    _test()
