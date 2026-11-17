"""
LC 4004. Minimum Moves to Balance Circular Array II — Hard
https://leetcode.com/problems/minimum-moves-to-balance-circular-array-ii/
Best: kamyu104 reference solution (verified on examples)
"""

# Time:  O(n^2 * logn)
# Space: O(n)

import heapq


# l1 isotonic regression, heap, greedy
class Solution(object):
    def minMoves(self, balance):
        """
        :type balance: List[int]
        :rtype: int
        """
        def clamp(x, l, r):
            return min(max(x, l), r)
    
        def cost(i):
            result = prefix = 0
            max_heap = []
            for j in range(len(balance)):
                prefix += balance[(i+j)%len(balance)]
                c = clamp(prefix, 0, total)
                result += abs(prefix-c)
                # heap-based l1 isotonic regression, reference: https://codeforces.com/contest/13/problem/C
                heapq.heappush(max_heap, -c)
                if -max_heap[0] > c:
                    result += -heapq.heappop(max_heap)-c
                    heapq.heappush(max_heap, -c)
            return result

        total = sum(balance)
        return min(cost(i) for i in range(len(balance))) if total >= 0 else -1


# Time:  O(F * E * logV) = O(n^2 * logn), V = O(n), E = O(n), and each augmentation saturates a supply or demand edge, so there are only O(n) augmentations
# Space: O(V + E) = O(n)
# min-cost max-flow, ssp, dijkstra's algorithm, johnson potential
import heapq


# Time: O(F * E * logV)
# Space: O(V + E)
# Template: https://github.com/kth-competitive-programming/kactl/blob/main/content/graph/MinCostMaxFlow.h
INF = float("inf")
class Edge(object):
    def __init__(self, from_node, to, rev, cap, cost, flow):
        self.from_node = from_node
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost
        self.flow = flow


class MCMF(object):
    def __init__(self, n):
        self.N = n
        self.ed = [[] for _ in range(n)]
        self.seen = [0]*n
        self.dist = [INF]*n
        self.pi = [0]*n
        self.par = [None]*n

    def addEdge(self, from_node, to, cap, cost):
        if from_node == to:
            return
        self.ed[from_node].append(Edge(from_node, to, len(self.ed[to]), cap, cost, 0))
        self.ed[to].append(Edge(to, from_node, len(self.ed[from_node])-1, 0, -cost, 0))

    def path(self, s):
        self.seen = [0]*self.N
        self.dist = [INF]*self.N
        self.par = [None]*self.N
        self.dist[s] = 0
        q = [(0, s)]
        while q:
            d, u = heapq.heappop(q)
            if d != self.dist[u]:
                continue
            self.seen[u] = 1
            for edge in self.ed[u]:
                if edge.cap-edge.flow <= 0:
                    continue
                val = d+self.pi[u]-self.pi[edge.to]+edge.cost
                if val < self.dist[edge.to]:
                    self.dist[edge.to] = val
                    self.par[edge.to] = edge
                    heapq.heappush(q, (val, edge.to))
        for i in range(self.N):
            if self.dist[i] != INF:
                self.pi[i] += self.dist[i]

    def maxflow(self, s, t):
        total_flow = total_cost = 0
        while True:
            self.path(s)
            if not self.seen[t]:
                break
            flow = INF
            edge = self.par[t]
            while edge:
                flow = min(flow, edge.cap-edge.flow)
                edge = self.par[edge.from_node]
            total_flow += flow
            edge = self.par[t]
            while edge:
                edge.flow += flow
                self.ed[edge.to][edge.rev].flow -= flow
                edge = self.par[edge.from_node]
        for edges in self.ed:
            for edge in edges:
                total_cost += edge.cost*edge.flow
        return total_flow, total_cost//2

def _test():
    s = Solution()
    assert s.minMoves(balance=[-1, 2, -1]) == 2
    assert s.minMoves(balance=[4, -1, -2]) == 3
    assert s.minMoves(balance=[-3, -3, 5]) == -1
    print("LC 4004 OK")


if __name__ == "__main__":
    _test()
