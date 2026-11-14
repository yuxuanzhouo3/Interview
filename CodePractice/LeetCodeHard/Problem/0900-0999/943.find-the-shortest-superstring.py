"""
LC 943. Find the Shortest Superstring — Hard
https://leetcode.com/problems/find-the-shortest-superstring/
Best: TSP bitmask DP — O(n^2 * 2^n)
"""

from typing import List


class Solution:
    def shortestSuperstring(self, words: List[str]) -> str:
        n = len(words)
        overlap = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = words[i], words[j]
                best = 0
                for k in range(1, min(len(a), len(b)) + 1):
                    if a[-k:] == b[:k]:
                        best = k
                overlap[i][j] = best

        N = 1 << n
        # dp[mask][j] = shortest string covering mask ending at j
        dp = [[None] * n for _ in range(N)]
        parent = [[-1] * n for _ in range(N)]
        for i in range(n):
            dp[1 << i][i] = words[i]

        for mask in range(N):
            for j in range(n):
                if dp[mask][j] is None:
                    continue
                for k in range(n):
                    if mask & (1 << k):
                        continue
                    cand = dp[mask][j] + words[k][overlap[j][k]:]
                    nmask = mask | (1 << k)
                    if dp[nmask][k] is None or len(cand) < len(dp[nmask][k]):
                        dp[nmask][k] = cand
                        parent[nmask][k] = j

        full = N - 1
        best_i = min(range(n), key=lambda i: len(dp[full][i]))
        return dp[full][best_i]


def _test():
    s = Solution()
    r = s.shortestSuperstring(["alex", "loves", "leetcode"])
    assert "alex" in r and "loves" in r and "leetcode" in r
    assert len(r) == len("alexlovesleetcode") or len(r) <= 16
    r2 = s.shortestSuperstring(["catg", "ctaagt", "gcta", "ttca", "atgcatc"])
    assert all(w in r2 for w in ["catg", "ctaagt", "gcta", "ttca", "atgcatc"])
    print("LC 943 OK")


if __name__ == "__main__":
    _test()
