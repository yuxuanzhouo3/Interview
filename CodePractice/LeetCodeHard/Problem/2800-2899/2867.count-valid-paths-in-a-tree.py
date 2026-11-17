"""
LC 2867. Count Valid Paths in a Tree — Hard
https://leetcode.com/problems/count-valid-paths-in-a-tree/
Best: Sieve primes + DFS non-prime components — O(n)
"""

from typing import List


class Solution:
    def countPaths(self, n: int, edges: List[List[int]]) -> int:
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        g = [[] for _ in range(n + 1)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        ans = 0
        for p in range(1, n + 1):
            if not is_prime[p]:
                continue
            comps = []
            for v in g[p]:
                if is_prime[v]:
                    continue
                stack = [v]
                seen = {v}
                sz = 0
                while stack:
                    u = stack.pop()
                    sz += 1
                    for w in g[u]:
                        if w == p or w in seen or is_prime[w]:
                            continue
                        seen.add(w)
                        stack.append(w)
                comps.append(sz)
            s = sum(comps)
            pairs = (s * s - sum(c * c for c in comps)) // 2
            ans += s + pairs
        return ans


def _test():
    s = Solution()
    assert s.countPaths(5, [[1, 2], [1, 3], [2, 4], [2, 5]]) == 4
    assert s.countPaths(6, [[1, 2], [1, 3], [2, 4], [3, 5], [3, 6]]) == 6
    print("LC 2867 OK")


if __name__ == "__main__":
    _test()
