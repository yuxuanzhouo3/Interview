"""
LC 1617. Count Subtrees With Max Distance Between Cities — Hard
https://leetcode.com/problems/count-subtrees-with-max-distance-between-cities/
Best: Enumerate subsets + BFS diameter — O(2^n · n^2)
"""

from collections import deque
from typing import List


class Solution:
    def countSubgraphsForEachDiameter(self, n: int, edges: List[List[int]]) -> List[int]:
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u - 1].append(v - 1)
            g[v - 1].append(u - 1)
        ans = [0] * (n - 1)

        def diameter(mask: int) -> int:
            nodes = [i for i in range(n) if mask & (1 << i)]
            if len(nodes) < 2:
                return -1
            start = nodes[0]
            q = deque([start])
            seen = {start}
            while q:
                u = q.popleft()
                for v in g[u]:
                    if mask & (1 << v) and v not in seen:
                        seen.add(v)
                        q.append(v)
            if len(seen) != len(nodes):
                return -1

            def farthest(src):
                q = deque([(src, 0)])
                vis = {src}
                far, dist = src, 0
                while q:
                    u, d = q.popleft()
                    if d > dist:
                        far, dist = u, d
                    for v in g[u]:
                        if mask & (1 << v) and v not in vis:
                            vis.add(v)
                            q.append((v, d + 1))
                return far, dist

            a, _ = farthest(nodes[0])
            _, d = farthest(a)
            return d

        for mask in range(1, 1 << n):
            d = diameter(mask)
            if d >= 1:
                ans[d - 1] += 1
        return ans


def _test():
    s = Solution()
    assert s.countSubgraphsForEachDiameter(4, [[1, 2], [2, 3], [2, 4]]) == [3, 4, 0]
    assert s.countSubgraphsForEachDiameter(2, [[1, 2]]) == [1]
    assert s.countSubgraphsForEachDiameter(3, [[1, 2], [2, 3]]) == [2, 1]
    print("LC 1617 OK")


if __name__ == "__main__":
    _test()
