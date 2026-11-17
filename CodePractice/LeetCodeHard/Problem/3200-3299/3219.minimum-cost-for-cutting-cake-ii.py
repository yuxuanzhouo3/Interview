"""
LC 3219. Minimum Cost for Cutting Cake II — Hard
https://leetcode.com/problems/minimum-cost-for-cutting-cake-ii/
Best: Greedy sort cuts descending — O(n log n)
"""

from typing import List


class Solution:
    def minimumCost(
        self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]
    ) -> int:
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        i = j = 0
        h_pieces = v_pieces = 1
        ans = 0
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] >= verticalCut[j]:
                ans += horizontalCut[i] * v_pieces
                h_pieces += 1
                i += 1
            else:
                ans += verticalCut[j] * h_pieces
                v_pieces += 1
                j += 1
        while i < len(horizontalCut):
            ans += horizontalCut[i] * v_pieces
            i += 1
        while j < len(verticalCut):
            ans += verticalCut[j] * h_pieces
            j += 1
        return ans


def _test():
    s = Solution()
    assert s.minimumCost(3, 2, [1, 3], [5]) == 13
    assert s.minimumCost(2, 2, [7], [4]) == 15
    print("LC 3219 OK")


if __name__ == "__main__":
    _test()
