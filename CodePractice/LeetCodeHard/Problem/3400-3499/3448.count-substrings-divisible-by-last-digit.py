"""
LC 3448. Count Substrings Divisible By Last Digit — Hard
https://leetcode.com/problems/count-substrings-divisible-by-last-digit/
Best: DP on prefix mod 1..9 — O(n * 9 * 9)
"""


class Solution:
    def countSubstrings(self, s: str) -> int:
        # count substrings where int(sub) % last_digit == 0 (last != 0)
        ans = 0
        # dp[m][r] = number of substrings ending here with value % m == r
        # Actually maintain for each modulus m=1..9 the count of prefixes of current ending
        # Restart approach: for each end digit d, need substring % d == 0
        # Maintain cnt[mod][rem] across the string as we extend
        cnt = [[0] * m for m in range(10)]  # unused index 0
        for ch in s:
            d = ord(ch) - 48
            ncnt = [[0] * m for m in range(10)]
            for m in range(1, 10):
                # new single digit
                ncnt[m][d % m] += 1
                for r in range(m):
                    if cnt[m][r]:
                        ncnt[m][(r * 10 + d) % m] += cnt[m][r]
            cnt = ncnt
            if d != 0:
                ans += cnt[d][0]
        return ans


def _test():
    s = Solution()
    assert s.countSubstrings("12936") == 11
    assert s.countSubstrings("5701283") == 18
    assert s.countSubstrings("1010101010") == 25
    print("LC 3448 OK")


if __name__ == "__main__":
    _test()
