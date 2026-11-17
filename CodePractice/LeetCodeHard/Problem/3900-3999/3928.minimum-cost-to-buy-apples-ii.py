"""
LC 3928. Minimum Cost to Buy Apples II — Hard
https://leetcode.com/problems/minimum-cost-to-buy-apples-ii/
Best: layered Dijkstra (empty / carrying) per start shop
"""
import heapq


class Solution:
    def minCost(self, n: int, prices: list[int], roads: list[list[int]]) -> list[int]:
        adj = [[] for _ in range(2 * n)]
        for u, v, w, tax in roads:
            adj[u].append((v, w))
            adj[v].append((u, w))
            adj[u + n].append((v + n, w * tax))
            adj[v + n].append((u + n, w * tax))

        def helper(start: int) -> int:
            INF = 10**18
            cost = [INF] * (2 * n)
            res = INF
            pq = []

            def push(idx: int, c: int):
                nonlocal res
                if c < cost[idx] and c < res:
                    cost[idx] = c
                    heapq.heappush(pq, (c, idx))
                    a = idx - n if idx >= n else idx
                    res = min(res, cost[a] + cost[a + n] + prices[a])

            push(start, 0)
            push(start + n, 0)
            while pq:
                c, u = heapq.heappop(pq)
                if c != cost[u]:
                    continue
                for v, w in adj[u]:
                    push(v, c + w)
            return res

        return [helper(i) for i in range(n)]


def _test():
    s = Solution()
    assert s.minCost(2, [8, 3], [[0, 1, 1, 2]]) == [6, 3]
    assert s.minCost(3, [9, 4, 6], [[0, 1, 1, 3], [1, 2, 4, 2]]) == [8, 4, 6]
    assert s.minCost(3, [10, 11, 1], [[0, 2, 1, 3], [1, 2, 3, 4], [0, 1, 5, 2]]) == [5, 11, 1]
    print("LC 3928 OK")


if __name__ == "__main__":
    _test()
