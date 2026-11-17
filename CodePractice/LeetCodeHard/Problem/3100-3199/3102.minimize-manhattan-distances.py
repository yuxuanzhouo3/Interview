"""
LC 3102. Minimize Manhattan Distances — Hard
https://leetcode.com/problems/minimize-manhattan-distances/
Best: Chebyshev extremes, try remove one extreme — O(n)
"""

from typing import List


class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        us = [x + y for x, y in points]
        vs = [x - y for x, y in points]
        cands = {
            max(range(n), key=lambda i: us[i]),
            min(range(n), key=lambda i: us[i]),
            max(range(n), key=lambda i: vs[i]),
            min(range(n), key=lambda i: vs[i]),
        }

        def dist(skip: int) -> int:
            maxu = maxv = float("-inf")
            minu = minv = float("inf")
            for i in range(n):
                if i == skip:
                    continue
                maxu = max(maxu, us[i])
                minu = min(minu, us[i])
                maxv = max(maxv, vs[i])
                minv = min(minv, vs[i])
            return max(maxu - minu, maxv - minv)

        return min(dist(i) for i in cands)


def _test():
    s = Solution()
    assert s.minimumDistance([[3, 10], [5, 15], [10, 2], [4, 4]]) == 12
    assert s.minimumDistance([[1, 1], [1, 1], [1, 1]]) == 0
    assert s.minimumDistance([[1, 1], [1, 2], [2, 1], [2, 2]]) == 2
    print("LC 3102 OK")


if __name__ == "__main__":
    _test()
