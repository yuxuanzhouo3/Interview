"""
LC 2127. Maximum Employees to Be Invited to a Meeting — Hard
https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/
Best: cycle + reverse graph chain — O(n)
"""

from typing import List
from collections import deque


class Solution:
    def maximumInvitations(self, favorite: List[int]) -> int:
        n = len(favorite)
        indeg = [0] * n
        for f in favorite:
            indeg[f] += 1
        # longest chain into each node (topo peel)
        depth = [1] * n
        q = deque(i for i in range(n) if indeg[i] == 0)
        while q:
            u = q.popleft()
            v = favorite[u]
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
        # remaining are cycles
        max_cycle = 0
        two_sum = 0
        seen = [False] * n
        for i in range(n):
            if indeg[i] == 0 or seen[i]:
                continue
            # walk cycle
            x = i
            length = 0
            while not seen[x]:
                seen[x] = True
                x = favorite[x]
                length += 1
            if length == 2:
                # mutual pair: add both chains
                a, b = i, favorite[i]
                two_sum += depth[a] + depth[b]
            else:
                max_cycle = max(max_cycle, length)
        return max(max_cycle, two_sum)


def _test():
    s = Solution()
    assert s.maximumInvitations([2, 2, 1, 2]) == 3
    assert s.maximumInvitations([1, 2, 0]) == 3
    assert s.maximumInvitations([3, 0, 1, 4, 1]) == 4
    print("LC 2127 OK")


if __name__ == "__main__":
    _test()
