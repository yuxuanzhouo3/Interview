"""
LC 1231. Divide Chocolate — Hard
https://leetcode.com/problems/divide-chocolate/
Tags: Binary Search
Best: 二分最小甜度 + 贪心切段
"""

from typing import List


class Solution:
    def maximizeSweetness(self, sweetness: List[int], K: int) -> int:
        # K friends => K+1 pieces; maximize min piece sum
        lo, hi = 1, sum(sweetness) // (K + 1)

        def can(x):
            s = cnt = 0
            for v in sweetness:
                s += v
                if s >= x:
                    cnt += 1
                    s = 0
            return cnt >= K + 1

        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


def _test():
    sol = Solution()
    assert sol.maximizeSweetness([1, 2, 3, 4, 5, 6, 7, 8, 9], 5) == 6
    assert sol.maximizeSweetness([5, 6, 7, 8, 9, 1, 2, 3, 4], 8) == 1
    assert sol.maximizeSweetness([1, 2, 2, 1, 2, 2, 1, 2, 2], 2) == 5
    print("LC 1231 OK")


if __name__ == "__main__":
    _test()

