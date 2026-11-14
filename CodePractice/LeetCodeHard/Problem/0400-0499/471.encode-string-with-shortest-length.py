"""
LC 471. Encode String with Shortest Length — Hard
https://leetcode.com/problems/encode-string-with-shortest-length/
Best: Interval DP + repeat detection — O(n^3)
"""


class Solution:
    def encode(self, s: str) -> str:
        n = len(s)
        if n <= 4:
            return s
        dp = [[""] * n for _ in range(n)]

        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                substr = s[i : j + 1]
                dp[i][j] = substr
                if length > 4:
                    for k in range(i, j):
                        left, right = dp[i][k], dp[k + 1][j]
                        if len(left) + len(right) < len(dp[i][j]):
                            dp[i][j] = left + right
                    # repeat pattern: substr = pattern * times
                    rep = (substr + substr).find(substr, 1)
                    if 0 < rep < length:
                        times = length // rep
                        encoded = f"{times}[{dp[i][i + rep - 1]}]"
                        if len(encoded) < len(dp[i][j]):
                            dp[i][j] = encoded
        return dp[0][n - 1]


def _test():
    s = Solution()
    assert s.encode("aaa") == "aaa"
    assert s.encode("aaaaa") in ("5[a]", "aaaaa")  # 5[a] shorter
    assert s.encode("aaaaa") == "5[a]"
    assert s.encode("aabcaabcd") == "2[aabc]d"
    assert s.encode("abbbabbbcabbbabbbc") == "2[2[abbb]c]"
    print("LC 471 OK")


if __name__ == "__main__":
    _test()
