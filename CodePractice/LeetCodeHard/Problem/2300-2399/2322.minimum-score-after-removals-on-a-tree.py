"""
LC 2322. Minimum Score After Removals on a Tree — Hard
https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/
Best: Tree XOR + enumerate 2 edge cuts — O(n^2)
"""


from typing import List
from collections import defaultdict


class Solution:
    def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
        n = len(nums)
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        xor = [0] * n
        tin = [0] * n
        tout = [0] * n
        timer = 0

        def dfs(u, p):
            nonlocal timer
            timer += 1
            tin[u] = timer
            x = nums[u]
            for v in g[u]:
                if v != p:
                    x ^= dfs(v, u)
            xor[u] = x
            tout[u] = timer
            return x

        dfs(0, -1)
        total = xor[0]

        def is_ancestor(a, b):
            return tin[a] <= tin[b] <= tout[a]

        # each node u!=0 corresponds to cutting edge to parent, component xor = xor[u]
        nodes = list(range(1, n))
        ans = 10**18
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if is_ancestor(a, b):
                    x1, x2, x3 = xor[b], xor[a] ^ xor[b], total ^ xor[a]
                elif is_ancestor(b, a):
                    x1, x2, x3 = xor[a], xor[b] ^ xor[a], total ^ xor[b]
                else:
                    x1, x2, x3 = xor[a], xor[b], total ^ xor[a] ^ xor[b]
                ans = min(ans, max(x1, x2, x3) - min(x1, x2, x3))
        return ans


def _test():
    s = Solution()
    assert s.minimumScore([1, 5, 5, 4, 11], [[0, 1], [1, 2], [1, 3], [3, 4]]) == 9
    assert s.minimumScore([5, 5, 2, 4, 4, 2], [[0, 1], [1, 2], [5, 2], [4, 3], [1, 3]]) == 0
    print("LC 2322 OK")


if __name__ == "__main__":
    _test()

