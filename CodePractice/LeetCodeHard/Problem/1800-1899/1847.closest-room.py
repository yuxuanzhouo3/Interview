"""
LC 1847. Closest Room — Hard
https://leetcode.com/problems/closest-room/
Best: Sort rooms/queries by size + SortedList id — O(n log n)
"""


from typing import List
from sortedcontainers import SortedList


class Solution:
    def closestRoom(self, rooms: List[List[int]], queries: List[List[int]]) -> List[int]:
        rooms.sort(key=lambda x: -x[1])
        qid = sorted(range(len(queries)), key=lambda i: -queries[i][1])
        ans = [-1] * len(queries)
        avail = SortedList()
        j = 0
        for i in qid:
            pref, minSize = queries[i]
            while j < len(rooms) and rooms[j][1] >= minSize:
                avail.add(rooms[j][0])
                j += 1
            if not avail:
                continue
            pos = avail.bisect_left(pref)
            best = None
            if pos < len(avail):
                best = avail[pos]
            if pos > 0:
                cand = avail[pos - 1]
                if best is None or abs(cand - pref) < abs(best - pref) or (
                    abs(cand - pref) == abs(best - pref) and cand < best
                ):
                    best = cand
            ans[i] = best
        return ans


def _test():
    s = Solution()
    assert s.closestRoom([[2, 2], [1, 2], [3, 2]], [[3, 1], [3, 3], [1, 2]]) == [3, -1, 1]
    assert s.closestRoom([[1, 4], [2, 3], [3, 5], [4, 1], [5, 2]], [[2, 3], [2, 4], [2, 5]]) == [2, 1, 3]
    print("LC 1847 OK")


if __name__ == "__main__":
    _test()

