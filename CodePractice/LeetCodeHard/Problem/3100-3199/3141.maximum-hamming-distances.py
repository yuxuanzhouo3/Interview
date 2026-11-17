"""
LC 3141. Maximum Hamming Distances — Hard
https://leetcode.com/problems/maximum-hamming-distances/
Best: Multi-source BFS on hypercube — O(m*2^m)
"""


from typing import List
from collections import deque


class Solution:
    def maxHammingDistances(self, nums: List[int], m: int) -> List[int]:
        N = 1 << m
        dist = [-1] * N
        q = deque()
        for x in nums:
            if dist[x] == -1:
                dist[x] = 0
                q.append(x)
        while q:
            u = q.popleft()
            for b in range(m):
                v = u ^ (1 << b)
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        mask = N - 1
        return [m - dist[mask ^ x] for x in nums]


def _test():
    s = Solution()
    assert s.maxHammingDistances([9, 12, 9, 11], 4) == [2, 3, 2, 3]
    r = s.maxHammingDistances([8, 8, 15], 4)
    assert r == [3, 3, 3]
    print("LC 3141 OK")


if __name__ == "__main__":
    _test()

