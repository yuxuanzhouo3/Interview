"""
LC 1203. Sort Items by Groups Respecting Dependencies — Hard
https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/
Best: Two-level topological sort — O(n + e)
"""

from collections import deque
from typing import List


class Solution:
    def sortItems(
        self, n: int, m: int, group: List[int], beforeItems: List[List[int]]
    ) -> List[int]:
        for i in range(n):
            if group[i] == -1:
                group[i] = m
                m += 1

        item_graph = [[] for _ in range(n)]
        item_indeg = [0] * n
        group_graph = [[] for _ in range(m)]
        group_indeg = [0] * m
        group_items = [[] for _ in range(m)]
        seen_g = set()

        for i in range(n):
            group_items[group[i]].append(i)

        for i, befores in enumerate(beforeItems):
            gi = group[i]
            for j in befores:
                item_graph[j].append(i)
                item_indeg[i] += 1
                gj = group[j]
                if gi != gj and (gj, gi) not in seen_g:
                    seen_g.add((gj, gi))
                    group_graph[gj].append(gi)
                    group_indeg[gi] += 1

        def topo(nodes, graph, indeg):
            deg = indeg[:]
            q = deque([u for u in nodes if deg[u] == 0])
            order = []
            while q:
                u = q.popleft()
                order.append(u)
                for v in graph[u]:
                    deg[v] -= 1
                    if deg[v] == 0:
                        q.append(v)
            return order if len(order) == len(nodes) else []

        g_order = topo(list(range(m)), group_graph, group_indeg)
        if not g_order:
            return []
        i_order = topo(list(range(n)), item_graph, item_indeg)
        if not i_order:
            return []

        pos = {x: i for i, x in enumerate(i_order)}
        ans = []
        for g in g_order:
            ans.extend(sorted(group_items[g], key=lambda x: pos[x]))
        return ans


def _test():
    s = Solution()
    n, m = 8, 2
    group = [-1, -1, 1, 0, 0, 1, 0, -1]
    before = [[], [6], [5], [6], [3, 6], [], [], []]
    res = s.sortItems(n, m, group[:], [b[:] for b in before])
    assert len(res) == n and set(res) == set(range(n))
    idx = {v: i for i, v in enumerate(res)}
    for i, befores in enumerate(before):
        for j in befores:
            assert idx[j] < idx[i]
    # same group contiguous
    g_assign = [-1, -1, 1, 0, 0, 1, 0, -1]
    last = {}
    seen_gap = set()
    for x in res:
        g = g_assign[x]
        if g == -1:
            continue
        if g in last and last[g] + 1 != idx[x] and g in seen_gap:
            # allow only contiguous block: once left group, cannot return
            pass
        last[g] = idx[x]
    # check contiguous groups
    blocks = {}
    for i, x in enumerate(res):
        g = g_assign[x]
        if g == -1:
            continue
        if g not in blocks:
            blocks[g] = [i, i]
        else:
            assert blocks[g][1] + 1 == i
            blocks[g][1] = i

    assert s.sortItems(8, 2, [-1, -1, 1, 0, 0, 1, 0, -1], [[], [6], [5], [6], [3], [], [4], []]) == []
    print("LC 1203 OK")


if __name__ == "__main__":
    _test()
