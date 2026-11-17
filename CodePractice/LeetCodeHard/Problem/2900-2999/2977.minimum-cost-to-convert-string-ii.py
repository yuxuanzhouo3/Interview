"""
LC 2977. Minimum Cost to Convert String II — Hard
Best: Dijkstra on positions + Aho-Corasick / DP — O(n^2 * m)
"""

from typing import List
import heapq


class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # build min cost map for string replacements
        from collections import defaultdict
        mp = defaultdict(lambda: float("inf"))
        nodes = set()
        for a, b, c in zip(original, changed, cost):
            mp[(a, b)] = min(mp[(a, b)], c)
            nodes.add(a); nodes.add(b)
        # floyd on distinct strings that appear
        words = list(nodes)
        # also need single-step only for matching substrings
        n = len(source)
        if source == target:
            return 0
        dist = [float("inf")] * (n + 1)
        dist[0] = 0
        for i in range(n):
            if dist[i] == float("inf"):
                continue
            if source[i] == target[i]:
                dist[i + 1] = min(dist[i + 1], dist[i])
            for a, b in list(mp.keys()):
                L = len(a)
                if i + L <= n and source[i:i + L] == a and target[i:i + L] == b:
                    dist[i + L] = min(dist[i + L], dist[i] + mp[(a, b)])
        # improve with floyd between intermediates - skip for basic
        # add transitive via Floyd on words then match
        idx = {w: i for i, w in enumerate(words)}
        m = len(words)
        g = [[float("inf")] * m for _ in range(m)]
        for i in range(m):
            g[i][i] = 0
        for (a, b), c in mp.items():
            g[idx[a]][idx[b]] = min(g[idx[a]][idx[b]], c)
        for k in range(m):
            for i in range(m):
                for j in range(m):
                    if g[i][k] + g[k][j] < g[i][j]:
                        g[i][j] = g[i][k] + g[k][j]
        dist = [float("inf")] * (n + 1)
        dist[0] = 0
        for i in range(n):
            if dist[i] == float("inf"):
                continue
            if source[i] == target[i]:
                dist[i + 1] = min(dist[i + 1], dist[i])
            for a in words:
                L = len(a)
                if i + L <= n and source[i:i + L] == a:
                    tb = target[i:i + L]
                    if tb in idx and g[idx[a]][idx[tb]] < float("inf"):
                        dist[i + L] = min(dist[i + L], dist[i] + g[idx[a]][idx[tb]])
        return -1 if dist[n] == float("inf") else int(dist[n])


def _test():
    s = Solution()
    assert s.minimumCost("abcd", "acbe", ["a", "b", "c", "c", "e", "d"], ["b", "c", "b", "e", "b", "e"], [2, 5, 5, 1, 2, 20]) == 28
    print("LC 2977 OK")


if __name__ == "__main__":
    _test()
