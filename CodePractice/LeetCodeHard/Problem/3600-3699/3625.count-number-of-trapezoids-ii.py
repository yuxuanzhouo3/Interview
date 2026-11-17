"""
LC 3625. Count Number of Trapezoids II — Hard
https://leetcode.com/problems/count-number-of-trapezoids-ii/
Best: doocs/leetcode reference solution (verified on examples)
"""

from collections import defaultdict, Counter


class Solution:
    def countTrapezoids(self, points: list[list[int]]) -> int:
        n = len(points)

        # cnt1: k -> (b -> count)
        cnt1: dict[float, dict[float, int]] = defaultdict(lambda: defaultdict(int))
        # cnt2: p -> (k -> count)
        cnt2: dict[int, dict[float, int]] = defaultdict(lambda: defaultdict(int))

        for i in range(n):
            x1, y1 = points[i]
            for j in range(i):
                x2, y2 = points[j]
                dx, dy = x2 - x1, y2 - y1

                if dx == 0:
                    k = 1e9
                    b = x1
                else:
                    k = dy / dx
                    b = (y1 * dx - x1 * dy) / dx

                cnt1[k][b] += 1

                p = (x1 + x2 + 2000) * 4001 + (y1 + y2 + 2000)
                cnt2[p][k] += 1

        ans = 0

        for e in cnt1.values():
            s = 0
            for t in e.values():
                ans += s * t
                s += t

        for e in cnt2.values():
            s = 0
            for t in e.values():
                ans -= s * t
                s += t

        return ans

def _test():
    s = Solution()
    assert s.countTrapezoids([[-3,2],[3,0],[2,3],[3,2],[2,-3]]) == 2
    assert s.countTrapezoids([[0,0],[1,0],[0,1],[2,1]]) == 1
    print("LC 3625 OK")


if __name__ == "__main__":
    _test()
