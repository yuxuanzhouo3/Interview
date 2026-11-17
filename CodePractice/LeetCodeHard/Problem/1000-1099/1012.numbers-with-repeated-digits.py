"""
LC 1012. Numbers With Repeated Digits — Hard
https://leetcode.com/problems/numbers-with-repeated-digits/
Best: Digit DP / count without repeats — O(log n)
"""


class Solution:
    def numDupDigitsAtMostN(self, n: int) -> int:
        s = str(n)

        def P(a: int, b: int) -> int:
            # a*(a-1)*...*(a-b+1)
            res = 1
            for i in range(b):
                res *= a - i
            return res

        # count numbers <= n with all unique digits, then n - that
        # 1) shorter lengths
        unique = 0
        d = len(s)
        for L in range(1, d):
            unique += 9 * P(9, L - 1)

        # 2) same length
        seen = set()
        for i, ch in enumerate(s):
            dig = int(ch)
            start = 1 if i == 0 else 0
            for x in range(start, dig):
                if x in seen:
                    continue
                unique += P(9 - i, d - i - 1)
            if dig in seen:
                break
            seen.add(dig)
        else:
            unique += 1  # n itself unique
        return n - unique


def _test():
    s = Solution()
    assert s.numDupDigitsAtMostN(20) == 1
    assert s.numDupDigitsAtMostN(100) == 10
    assert s.numDupDigitsAtMostN(1000) == 262
    print("LC 1012 OK")


if __name__ == "__main__":
    _test()
