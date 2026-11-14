"""
LC 1606. Find Servers That Handled Most Number of Requests — Hard
https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests/
Best: free ordered list + busy min-heap — O(n log k)
"""

from typing import List
from heapq import heappush, heappop
import bisect


class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        free = list(range(k))
        busy = []  # (free_time, server)
        cnt = [0] * k
        for i, (t, l) in enumerate(zip(arrival, load)):
            while busy and busy[0][0] <= t:
                _, s = heappop(busy)
                bisect.insort(free, s)
            if not free:
                continue
            target = i % k
            idx = bisect.bisect_left(free, target)
            if idx == len(free):
                idx = 0
            s = free.pop(idx)
            cnt[s] += 1
            heappush(busy, (t + l, s))
        mx = max(cnt)
        return [i for i, c in enumerate(cnt) if c == mx]


def _test():
    s = Solution()
    assert s.busiestServers(3, [1, 2, 3, 4, 5], [5, 2, 3, 3, 3]) == [1]
    assert s.busiestServers(3, [1, 2, 3, 4], [1, 2, 1, 2]) == [0]
    assert s.busiestServers(3, [1, 2, 3], [10, 12, 11]) == [0, 1, 2]
    print("LC 1606 OK")


if __name__ == "__main__":
    _test()
