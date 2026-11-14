"""
LC 332. Reconstruct Itinerary — Hard
Best: Hierholzer Euler path — O(E log E)
"""

from typing import List
from collections import defaultdict
import heapq


class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = defaultdict(list)
        for a, b in tickets:
            heapq.heappush(graph[a], b)
        route = []

        def dfs(u):
            while graph[u]:
                dfs(heapq.heappop(graph[u]))
            route.append(u)

        dfs("JFK")
        return route[::-1]


def _test():
    s = Solution()
    assert s.findItinerary([["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]) == [
        "JFK", "MUC", "LHR", "SFO", "SJC"
    ]
    print("LC 332 OK")


if __name__ == "__main__":
    _test()
