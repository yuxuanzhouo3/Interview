"""
LC 815. Bus Routes — Hard
Best: BFS on stops/buses — O(sum route lengths)
"""

from typing import List
from collections import defaultdict, deque


class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        if source == target:
            return 0
        stop_to_buses = defaultdict(list)
        for i, r in enumerate(routes):
            for s in r:
                stop_to_buses[s].append(i)
        q = deque([(source, 0)])
        seen_stop = {source}
        seen_bus = set()
        while q:
            stop, dist = q.popleft()
            for b in stop_to_buses[stop]:
                if b in seen_bus:
                    continue
                seen_bus.add(b)
                for nxt in routes[b]:
                    if nxt == target:
                        return dist + 1
                    if nxt not in seen_stop:
                        seen_stop.add(nxt)
                        q.append((nxt, dist + 1))
        return -1


def _test():
    s = Solution()
    assert s.numBusesToDestination([[1, 2, 7], [3, 6, 7]], 1, 6) == 2
    assert s.numBusesToDestination([[7, 12], [4, 5, 15], [6], [15, 19], [9, 12, 13]], 15, 12) == -1
    print("LC 815 OK")


if __name__ == "__main__":
    _test()
