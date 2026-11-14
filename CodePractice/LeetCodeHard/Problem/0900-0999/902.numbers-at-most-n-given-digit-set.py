"""
LC 902. Numbers At Most N Given Digit Set — Hard
https://leetcode.com/problems/numbers-at-most-n-given-digit-set/
Best: Digit DP — O(D * |digits|)
"""

from typing import List


class Solution:
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        s = str(n)
        D = len(s)
        digs = sorted(digits)
        m = len(digs)
        # shorter lengths
        ans = sum(m ** L for L in range(1, D))
        # same length, digit DP tight prefix
        for i, ch in enumerate(s):
            for d in digs:
                if d < ch:
                    ans += m ** (D - i - 1)
                elif d == ch:
                    break
            else:
                return ans
            if ch not in digs:
                return ans
        return ans + 1  # n itself formed


def _test():
    s = Solution()
    assert s.atMostNGivenDigitSet(["1", "3", "5", "7"], 100) == 20
    assert s.atMostNGivenDigitSet(["1", "4", "9"], 1000000000) == 29523
    assert s.atMostNGivenDigitSet(["7"], 8) == 1
    print("LC 902 OK")


if __name__ == "__main__":
    _test()
