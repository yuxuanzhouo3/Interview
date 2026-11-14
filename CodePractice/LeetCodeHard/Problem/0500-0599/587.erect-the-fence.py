"""
LC 587. Erect the Fence — Hard
https://leetcode.com/problems/erect-the-fence/
Best: Andrew monotone chain convex hull — O(n log n)
"""

from typing import List


class Solution:
    def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
        pts = sorted(map(tuple, trees))
        if len(pts) <= 1:
            return [list(p) for p in pts]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower, upper = [], []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
                lower.pop()
            lower.append(p)
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
                upper.pop()
            upper.append(p)

        hull = list(dict.fromkeys(lower[:-1] + upper[:-1]))
        return [list(p) for p in hull]


def _test():
    s = Solution()
    r = s.outerTrees([[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]])
    assert sorted(map(tuple, r)) == sorted([(1, 1), (2, 0), (4, 2), (3, 3), (2, 4)])
    r2 = s.outerTrees([[1, 2], [2, 2], [4, 2]])
    assert sorted(map(tuple, r2)) == sorted([(1, 2), (2, 2), (4, 2)])
    print("LC 587 OK")


if __name__ == "__main__":
    _test()
