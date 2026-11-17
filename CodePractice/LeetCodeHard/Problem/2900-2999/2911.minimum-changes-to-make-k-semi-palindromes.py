"""
LC 2911. Minimum Changes to Make K Semi-palindromes — Hard
https://leetcode.com/problems/minimum-changes-to-make-k-semi-palindromes/
Best: Cost table + partition DP — O(n^2 * d(n) + n^2 k)
"""


class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        # cost[i][j] = min changes to make s[i:j+1] a semi-palindrome
        # semi-palindrome: length >= 2, exists d|len, d<len, such that for each residue r,
        # the subsequence s[r],s[r+d],... is palindrome
        INF = 10**9
        cost = [[INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):  # len >= 2
                length = j - i + 1
                best = INF
                # divisors d of length with d < length
                d = 1
                while d * d <= length:
                    if length % d == 0:
                        for step in {d, length // d}:
                            if step < length:
                                # check residues 0..step-1
                                chg = 0
                                for r in range(step):
                                    # sequence indices i+r, i+r+step, ...
                                    seq = []
                                    p = i + r
                                    while p <= j:
                                        seq.append(s[p])
                                        p += step
                                    L, R = 0, len(seq) - 1
                                    while L < R:
                                        if seq[L] != seq[R]:
                                            chg += 1
                                        L += 1
                                        R -= 1
                                best = min(best, chg)
                    d += 1
                cost[i][j] = best
        # dp[c][i] = min cost to partition s[:i] into c parts
        dp = [[INF] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 0
        for c in range(1, k + 1):
            for i in range(2 * c, n + 1):  # each part len>=2
                for p in range(2 * (c - 1), i - 1):
                    if dp[c - 1][p] < INF and cost[p][i - 1] < INF:
                        dp[c][i] = min(dp[c][i], dp[c - 1][p] + cost[p][i - 1])
        return dp[k][n]


def _test():
    s = Solution()
    assert s.minimumChanges("abcac", 2) == 1
    assert s.minimumChanges("abcdef", 2) == 2
    assert s.minimumChanges("aabbaa", 3) == 0
    print("LC 2911 OK")


if __name__ == "__main__":
    _test()
