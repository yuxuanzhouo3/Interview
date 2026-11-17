"""
LC 3981. Count Distinct Ways to Form Target from Two Strings — Hard
https://leetcode.com/problems/count-distinct-ways-to-form-target-from-two-strings/
Best: DP on (i in target, p1, p2, used1, used2) with next-occurrence jumps
"""
MOD = 10**9 + 7


class Solution:
    def interleaveCharacters(self, word1: str, word2: str, target: str) -> int:
        n1, n2, m = len(word1), len(word2), len(target)
        # next occurrence of char from position
        nxt1 = [[n1] * 26 for _ in range(n1 + 1)]
        nxt2 = [[n2] * 26 for _ in range(n2 + 1)]
        for i in range(n1 - 1, -1, -1):
            nxt1[i] = nxt1[i + 1][:]
            nxt1[i][ord(word1[i]) - 97] = i
        for i in range(n2 - 1, -1, -1):
            nxt2[i] = nxt2[i + 1][:]
            nxt2[i][ord(word2[i]) - 97] = i

        from functools import lru_cache

        @lru_cache(None)
        def dfs(i: int, p1: int, p2: int, u1: int, u2: int) -> int:
            if i == m:
                return 1 if u1 and u2 else 0
            ch = ord(target[i]) - 97
            ans = 0
            j1 = nxt1[p1][ch]
            if j1 < n1:
                ans += dfs(i + 1, j1 + 1, p2, 1, u2)
            j2 = nxt2[p2][ch]
            if j2 < n2:
                ans += dfs(i + 1, p1, j2 + 1, u1, 1)
            return ans % MOD

        return dfs(0, 0, 0, 0, 0)


def _test():
    s = Solution()
    assert s.interleaveCharacters("abc", "bac", "abc") == 5
    assert s.interleaveCharacters("cd", "cd", "ccd") == 4
    assert s.interleaveCharacters("xy", "xy", "xyxy") == 2
    assert s.interleaveCharacters("ab", "cde", "ace") == 1
    print("LC 3981 OK")


if __name__ == "__main__":
    _test()
