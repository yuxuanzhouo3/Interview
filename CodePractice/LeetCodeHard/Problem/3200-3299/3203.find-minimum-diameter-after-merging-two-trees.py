"""
LC 3203. Find Minimum Diameter After Merging Two Trees — Hard
https://leetcode.com/problems/find-minimum-diameter-after-merging-two-trees/
Best: Tree diameters + radii connect — O(n)
"""

from typing import List
from collections import defaultdict, deque


class Solution:
    def minimumDiameterAfterMerge(
        self, edges1: List[List[int]], edges2: List[List[int]]
    ) -> int:
        def diameter(edges: List[List[int]]) -> int:
            if not edges:
                return 0
            g = defaultdict(list)
            for a, b in edges:
                g[a].append(b)
                g[b].append(a)

            def farthest(src: int):
                dist = {src: 0}
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in g[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            q.append(v)
                node = max(dist, key=dist.get)
                return node, dist[node]

            a, _ = farthest(0)
            _, d = farthest(a)
            return d

        d1 = diameter(edges1)
        d2 = diameter(edges2)
        return max(d1, d2, (d1 + 1) // 2 + (d2 + 1) // 2 + 1)


def _test():
    s = Solution()
    assert s.minimumDiameterAfterMerge([[0, 1], [0, 2], [0, 3]], [[0, 1]]) == 3
    assert (
        s.minimumDiameterAfterMerge(
            [[0, 1], [0, 2], [0, 3], [2, 4], [2, 5], [3, 6], [2, 7]],
            [[0, 1], [0, 2], [0, 3], [2, 4], [2, 5], [3, 6], [2, 7]],
        )
        == 5
    )
    print("LC 3203 OK")


if __name__ == "__main__":
    _test()
