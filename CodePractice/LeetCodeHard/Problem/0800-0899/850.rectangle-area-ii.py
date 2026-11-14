"""
LC 850. Rectangle Area II — Hard
Best: Sweep line + active y segments — O(n^2 log n)
"""

from typing import List

MOD = 10**9 + 7


class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        events = []
        for x1, y1, x2, y2 in rectangles:
            events.append((x1, 1, y1, y2))
            events.append((x2, -1, y1, y2))
        events.sort()
        active = []
        ans = 0
        prev_x = events[0][0]

        def cover_y():
            if not active:
                return 0
            segs = sorted(active)
            total = 0
            cur_s, cur_e = segs[0]
            for s, e in segs[1:]:
                if s > cur_e:
                    total += cur_e - cur_s
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            total += cur_e - cur_s
            return total

        for x, typ, y1, y2 in events:
            ans += cover_y() * (x - prev_x)
            if typ == 1:
                active.append((y1, y2))
            else:
                active.remove((y1, y2))
            prev_x = x
        return ans % MOD


def _test():
    s = Solution()
    assert s.rectangleArea([[0, 0, 2, 2], [1, 0, 2, 3], [1, 0, 3, 1]]) == 6
    print("LC 850 OK")


if __name__ == "__main__":
    _test()
