"""
LC 1956. Minimum Time For K Virus Variants to Spread — Hard
https://leetcode.com/problems/minimum-time-for-k-virus-variants-to-spread/
Best: 二分天数 + 旋转坐标扫线 — O(n^2 log n log T)
"""

from typing import List


class Solution:
    def minDayskVariants(self, points: List[List[int]], k: int) -> int:
        P = [(2 * x - 2 * y, 2 * x + 2 * y) for x, y in points]

        def check(m: int) -> bool:
            ys = sorted({y - m for _, y in P} | {y + m for _, y in P})
            yi = {v: i for i, v in enumerate(ys)}
            count = [0] * len(ys)
            events = []
            for x, y in P:
                events.append((x - m, 1, y - m, y + m))
                events.append((x + m, -1, y - m, y + m))
            events.sort(key=lambda e: (e[0], -e[1]))
            best = 0
            for _, op, y1, y2 in events:
                for i in range(yi[y1], yi[y2]):
                    count[i] += op
                    if count[i] > best:
                        best = count[i]
                if best >= k:
                    return True
            return False

        beg, end = -1, 400
        while beg + 1 < end:
            mid = (beg + end) // 2
            if check(mid):
                end = mid
            else:
                beg = mid
        return end // 2


def _test():
    s = Solution()
    assert s.minDayskVariants([[1, 1], [6, 1]], 2) == 3
    assert s.minDayskVariants([[3, 3], [1, 2], [9, 2]], 2) == 2
    assert s.minDayskVariants([[3, 3], [1, 2], [9, 2]], 3) == 4
    print("LC 1956 OK")


if __name__ == "__main__":
    _test()
