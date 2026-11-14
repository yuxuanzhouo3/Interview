"""
LC 269. Alien Dictionary — Hard
https://leetcode.com/problems/alien-dictionary/
Best: 比较相邻词建图 + 拓扑排序 — O(C)
"""

from typing import List
from collections import defaultdict, deque


class Solution:
    def alienOrder(self, words: List[str]) -> str:
        graph = defaultdict(set)
        indeg = {c: 0 for w in words for c in w}

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if len(w1) > len(w2) and w1.startswith(w2):
                return ""  # 非法前缀
            for a, b in zip(w1, w2):
                if a != b:
                    if b not in graph[a]:
                        graph[a].add(b)
                        indeg[b] += 1
                    break

        q = deque([c for c in indeg if indeg[c] == 0])
        order = []
        while q:
            c = q.popleft()
            order.append(c)
            for nei in graph[c]:
                indeg[nei] -= 1
                if indeg[nei] == 0:
                    q.append(nei)
        if len(order) != len(indeg):
            return ""
        return "".join(order)


def _test():
    s = Solution()
    r = s.alienOrder(["wrt", "wrf", "er", "ett", "rftt"])
    assert r == "wertf"
    r = s.alienOrder(["z", "x"])
    assert r == "zx"
    assert s.alienOrder(["z", "x", "z"]) == ""
    assert s.alienOrder(["abc", "ab"]) == ""
    print("LC 269 OK")


if __name__ == "__main__":
    _test()
