"""
LC 218. The Skyline Problem — Hard
Best: Sweep line + multiset of heights — O(n log n)
"""

from typing import List


class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # Prefer stdlib-only: heap with lazy delete
        import heapq
        events = []
        for L, R, H in buildings:
            events.append((L, -H, R))
            events.append((R, 0, 0))
        events.sort()
        # heap: (-height, end)
        live = [(0, float("inf"))]
        ans = []
        for x, negH, R in events:
            if negH:
                heapq.heappush(live, (negH, R))
            while live[0][1] <= x:
                heapq.heappop(live)
            h = -live[0][0]
            if not ans or ans[-1][1] != h:
                ans.append([x, h])
        return ans


def _test():
    s = Solution()
    b = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
    assert s.getSkyline(b) == [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]]
    print("LC 218 OK")


if __name__ == "__main__":
    _test()
