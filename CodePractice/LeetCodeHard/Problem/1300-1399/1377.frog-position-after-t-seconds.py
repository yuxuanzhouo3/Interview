"""
LC 1377. Frog Position After T Seconds — Hard
https://leetcode.com/problems/frog-position-after-t-seconds/
Best: DFS/BFS probability on tree — O(n)
"""

from collections import defaultdict
from typing import List


class Solution:
    def frogPosition(self, n: int, edges: List[List[int]], t: int, target: int) -> float:
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        # BFS
        from collections import deque

        q = deque([(1, 1.0, 0)])  # node, prob, time
        seen = {1}
        while q:
            u, p, time = q.popleft()
            children = [v for v in g[u] if v not in seen]
            if time == t or not children:
                if u == target:
                    return p
                continue
            for v in children:
                seen.add(v)
                q.append((v, p / len(children), time + 1))
        return 0.0


def _test():
    s = Solution()
    assert abs(s.frogPosition(7, [[1, 2], [1, 3], [1, 7], [2, 4], [2, 6], [3, 5]], 2, 4) - 0.16666666666666666) < 1e-9
    assert abs(s.frogPosition(7, [[1, 2], [1, 3], [1, 7], [2, 4], [2, 6], [3, 5]], 1, 7) - 0.3333333333333333) < 1e-9
    assert s.frogPosition(7, [[1, 2], [1, 3], [1, 7], [2, 4], [2, 6], [3, 5]], 20, 6) == 0.16666666666666666
    print("LC 1377 OK")


if __name__ == "__main__":
    _test()
