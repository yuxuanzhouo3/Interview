"""
LC 1092. Shortest Common Supersequence — Hard
https://leetcode.com/problems/shortest-common-supersequence/
Best: LCS DP + reconstruct — O(mn)
"""


class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        i, j = m, n
        chars = []
        while i > 0 and j > 0:
            if str1[i - 1] == str2[j - 1]:
                chars.append(str1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                chars.append(str1[i - 1])
                i -= 1
            else:
                chars.append(str2[j - 1])
                j -= 1
        while i > 0:
            chars.append(str1[i - 1])
            i -= 1
        while j > 0:
            chars.append(str2[j - 1])
            j -= 1
        return "".join(reversed(chars))


def _is_scs(scs, a, b):
    def is_subseq(s, t):
        i = 0
        for ch in t:
            if i < len(s) and s[i] == ch:
                i += 1
        return i == len(s)

    return is_subseq(a, scs) and is_subseq(b, scs)


def _test():
    s = Solution()
    r = s.shortestCommonSupersequence("abac", "cab")
    assert len(r) == 5 and _is_scs(r, "abac", "cab")
    r2 = s.shortestCommonSupersequence("aaaaaaaa", "aaaaaaaa")
    assert r2 == "aaaaaaaa"
    print("LC 1092 OK")


if __name__ == "__main__":
    _test()
