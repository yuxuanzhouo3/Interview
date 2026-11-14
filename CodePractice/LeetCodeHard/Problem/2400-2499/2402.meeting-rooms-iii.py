"""
LC 2402. Meeting Rooms III — Hard
https://leetcode.com/problems/meeting-rooms-iii/
Best: Two heaps (free rooms + busy) — O(m log m + m log n)
"""

from typing import List
import heapq


class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        free = list(range(n))
        heapq.heapify(free)
        busy = []  # (end, room)
        cnt = [0] * n
        for start, end in meetings:
            while busy and busy[0][0] <= start:
                _, r = heapq.heappop(busy)
                heapq.heappush(free, r)
            dur = end - start
            if free:
                r = heapq.heappop(free)
                heapq.heappush(busy, (end, r))
            else:
                avail, r = heapq.heappop(busy)
                heapq.heappush(busy, (avail + dur, r))
            cnt[r] += 1
        return max(range(n), key=lambda i: (cnt[i], -i))


def _test():
    s = Solution()
    assert s.mostBooked(2, [[0, 10], [1, 5], [2, 7], [3, 4]]) == 0
    assert s.mostBooked(3, [[1, 20], [2, 10], [3, 5], [4, 9], [6, 8]]) == 1
    print("LC 2402 OK")


if __name__ == "__main__":
    _test()
