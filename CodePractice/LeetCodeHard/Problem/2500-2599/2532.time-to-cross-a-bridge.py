"""
LC 2532. Time to Cross a Bridge — Hard
https://leetcode.com/problems/time-to-cross-a-bridge/
Best: Priority queues simulate bridge crossing — O(n log k)
"""

from heapq import heappush, heappop
from math import inf
from typing import List

class Solution:
    def findCrossingTime(self, n: int, k: int, time: List[List[int]]) -> int:
        time.sort(key=lambda x: x[0] + x[2])
        cur = 0
        wait_in_left, wait_in_right = [], []
        work_in_left, work_in_right = [], []
        for i in range(k):
            heappush(wait_in_left, -i)
        while True:
            while work_in_left:
                t, i = work_in_left[0]
                if t > cur:
                    break
                heappop(work_in_left)
                heappush(wait_in_left, -i)
            while work_in_right:
                t, i = work_in_right[0]
                if t > cur:
                    break
                heappop(work_in_right)
                heappush(wait_in_right, -i)
            left_to_go = n > 0 and wait_in_left
            right_to_go = bool(wait_in_right)
            if not left_to_go and not right_to_go:
                nxt = inf
                if work_in_left:
                    nxt = min(nxt, work_in_left[0][0])
                if work_in_right:
                    nxt = min(nxt, work_in_right[0][0])
                cur = nxt
                continue
            if right_to_go:
                i = -heappop(wait_in_right)
                cur += time[i][2]
                if n == 0 and not wait_in_right and not work_in_right:
                    return cur
                heappush(work_in_left, (cur + time[i][3], i))
            else:
                i = -heappop(wait_in_left)
                cur += time[i][0]
                n -= 1
                heappush(work_in_right, (cur + time[i][1], i))

def _test():

    s = Solution()
    assert s.findCrossingTime(1, 3, [[1, 1, 2, 1], [1, 1, 3, 1], [1, 1, 4, 1]]) == 6
    assert s.findCrossingTime(3, 2, [[1, 9, 1, 8], [10, 10, 10, 10]]) == 50
    print("LC 2532 OK")

if __name__ == "__main__":
    _test()
