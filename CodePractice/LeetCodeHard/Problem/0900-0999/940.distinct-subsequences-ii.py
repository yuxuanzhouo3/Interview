"""
LC 940. Distinct Subsequences II — Hard
https://leetcode.com/problems/distinct-subsequences-ii/
Best: DP last occurrence — O(n)
"""

class Solution:
    def distinctSubseqII(self, s: str) -> int:
        MOD = 10**9 + 7
        # endswith[c] = number of distinct nonempty subsequences ending with c
        endswith = [0] * 26
        for ch in s:
            i = ord(ch) - 97
            # all previous subsequences + empty, append ch; replaces old ending with ch
            total = (sum(endswith) + 1) % MOD
            endswith[i] = total
        return sum(endswith) % MOD


def _test():
    s = Solution()
    assert s.distinctSubseqII("abc") == 7
    assert s.distinctSubseqII("aba") == 6
    assert s.distinctSubseqII("aaa") == 3
    print("LC 940 OK")


if __name__ == "__main__":
    _test()
