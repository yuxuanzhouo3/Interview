"""
LC 2167. Minimum Time to Remove All Cars Containing Illegal Goods — Hard
https://leetcode.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/
Best: one-pass left DP + right suffix cost — O(n)
"""


class Solution:
    def minimumTime(self, s: str) -> int:
        n = len(s)
        # dp = min cost to clear illegal in prefix s[:i+1]
        # '1' can be removed alone (+2) or by deleting whole prefix (i+1)
        dp = 0
        ans = n  # delete all from one side
        for i, c in enumerate(s):
            if c == "1":
                dp = min(dp + 2, i + 1)
            ans = min(ans, dp + (n - 1 - i))  # + delete remaining from right
        return ans


def _test():
    s = Solution()
    assert s.minimumTime("1100101") == 5
    assert s.minimumTime("0010") == 2
    assert s.minimumTime("111") == 3
    print("LC 2167 OK")


if __name__ == "__main__":
    _test()
