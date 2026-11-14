"""
LC 660. Remove 9 — Hard
https://leetcode.com/problems/remove-9/
Best: map n to base-9 digits — O(log n)
"""


class Solution:
    def newInteger(self, n: int) -> int:
        # numbers without digit 9 are just base-9 numbers written in decimal glyphs 0-8
        digits = []
        while n:
            digits.append(str(n % 9))
            n //= 9
        return int("".join(reversed(digits)) or "0")


def _test():
    s = Solution()
    assert s.newInteger(9) == 10
    assert s.newInteger(10) == 11
    assert s.newInteger(1) == 1
    print("LC 660 OK")


if __name__ == "__main__":
    _test()
