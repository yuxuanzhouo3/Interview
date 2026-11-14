"""
LC 2246. Longest Path With Different Adjacent Characters — Hard
https://leetcode.com/problems/longest-path-with-different-adjacent-characters/
Best: Tree DFS — keep top-2 child paths with different labels — O(n)
"""

from typing import List


class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        n = len(parent)
        g = [[] for _ in range(n)]
        for i in range(1, n):
            g[parent[i]].append(i)
        ans = 1

        def dfs(u: int) -> int:
            nonlocal ans
            best1 = best2 = 0
            for v in g[u]:
                down = dfs(v)
                if s[v] == s[u]:
                    continue
                if down > best1:
                    best2, best1 = best1, down
                elif down > best2:
                    best2 = down
            ans = max(ans, best1 + best2 + 1)
            return best1 + 1

        dfs(0)
        return ans


def _test():
    s = Solution()
    assert s.longestPath([-1, 0, 0, 1, 1, 2], "abacbe") == 3
    assert s.longestPath([-1, 0, 0, 0], "aabc") == 3
    print("LC 2246 OK")


if __name__ == "__main__":
    _test()
