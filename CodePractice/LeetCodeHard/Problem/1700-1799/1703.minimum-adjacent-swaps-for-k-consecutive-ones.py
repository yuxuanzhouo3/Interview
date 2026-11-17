"""
LC 1703. Minimum Adjacent Swaps for K Consecutive Ones — Hard
https://leetcode.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/
Best: positions of 1s + median window + prefix — O(n)
"""

from typing import List


class Solution:
    def minMoves(self, nums: List[int], k: int) -> int:
        pos = [i for i, x in enumerate(nums) if x == 1]
        # cost to gather to median using prefix sums on (pos[i]-i)
        # equivalent: min swaps to make k ones consecutive = gather to consecutive positions
        n = len(pos)
        # g[i] = pos[i] - i  (normalized)
        g = [pos[i] - i for i in range(n)]
        pref = [0]
        for x in g:
            pref.append(pref[-1] + x)

        def range_sum(l, r):
            return pref[r + 1] - pref[l]

        best = float("inf")
        for i in range(n - k + 1):
            mid = i + k // 2
            # cost to move all to g[mid]
            left = g[mid] * (mid - i) - range_sum(i, mid - 1)
            right = range_sum(mid + 1, i + k - 1) - g[mid] * (i + k - 1 - mid)
            best = min(best, left + right)
        return best


def _test():
    s = Solution()
    assert s.minMoves([1, 0, 0, 1, 0, 1], 2) == 1
    assert s.minMoves([1, 0, 0, 0, 0, 0, 1, 1], 3) == 5
    assert s.minMoves([1, 1, 0, 1], 2) == 0
    print("LC 1703 OK")


if __name__ == "__main__":
    _test()
