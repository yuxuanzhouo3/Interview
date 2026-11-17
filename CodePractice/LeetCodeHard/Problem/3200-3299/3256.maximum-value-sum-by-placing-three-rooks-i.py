"""
LC 3256. Maximum Value Sum by Placing Three Rooks I — Hard
https://leetcode.com/problems/maximum-value-sum-by-placing-three-rooks-i/
Best: Per-row top3 + enumerate 3 rows — O(m n + m^3)
"""

from typing import List
import heapq


class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        # for each row, keep top 3 (value, col)
        tops = []
        for i, row in enumerate(board):
            tops.append(heapq.nlargest(3, ((v, j) for j, v in enumerate(row))))
        ans = -(10**18)
        # enumerate 3 distinct rows
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    # pick one cell from each row's top3 with distinct cols
                    for a, ca in tops[i]:
                        for b, cb in tops[j]:
                            if cb == ca:
                                continue
                            for c, cc in tops[k]:
                                if cc == ca or cc == cb:
                                    continue
                                ans = max(ans, a + b + c)
        return ans


def _test():
    s = Solution()
    assert s.maximumValueSum([[-3, 1, 1, 1], [-3, 1, -3, 1], [-3, 2, 1, 1]]) == 4
    assert s.maximumValueSum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 15
    assert s.maximumValueSum([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == 3
    print("LC 3256 OK")


if __name__ == "__main__":
    _test()
