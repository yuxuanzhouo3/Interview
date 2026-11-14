"""
LC 639. Decode Ways II — Hard
https://leetcode.com/problems/decode-ways-ii/
Best: DP with '*' cases — O(n)
"""

MOD = 10**9 + 7


class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)

        def one(c: str) -> int:
            if c == "*":
                return 9
            if c == "0":
                return 0
            return 1

        def two(a: str, b: str) -> int:
            if a == "*" and b == "*":
                return 15  # 11-19, 21-26
            if a == "*":
                if "0" <= b <= "6":
                    return 2  # 1x, 2x
                return 1  # 1x only
            if b == "*":
                if a == "1":
                    return 9
                if a == "2":
                    return 6
                return 0
            num = int(a) * 10 + int(b)
            return 1 if 10 <= num <= 26 else 0

        prev2, prev1 = 1, one(s[0])  # ways for empty, s[:1]
        for i in range(1, n):
            cur = (one(s[i]) * prev1 + two(s[i - 1], s[i]) * prev2) % MOD
            prev2, prev1 = prev1, cur
        return prev1 % MOD


def _test():
    s = Solution()
    assert s.numDecodings("*") == 9
    assert s.numDecodings("1*") == 18
    assert s.numDecodings("2*") == 15
    assert s.numDecodings("**") == 96
    print("LC 639 OK")


if __name__ == "__main__":
    _test()
