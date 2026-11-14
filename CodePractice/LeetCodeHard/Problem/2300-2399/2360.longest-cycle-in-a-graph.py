"""
LC 2360. Longest Cycle in a Graph — Hard
https://leetcode.com/problems/longest-cycle-in-a-graph/
Best: DFS with timestamps — each node outdegree ≤1 — O(n)
"""

from typing import List


class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        n = len(edges)
        time = [0] * n
        clock = 1
        ans = -1
        for i in range(n):
            if time[i]:
                continue
            start = clock
            u = i
            while u != -1 and time[u] == 0:
                time[u] = clock
                clock += 1
                u = edges[u]
            if u != -1 and time[u] >= start:
                ans = max(ans, clock - time[u])
        return ans


def _test():
    s = Solution()
    assert s.longestCycle([3, 3, 4, 2, 3]) == 3
    assert s.longestCycle([2, -1, 3, 1]) == -1
    print("LC 2360 OK")


if __name__ == "__main__":
    _test()
