"""
LC 972. Equal Rational Numbers — Hard
https://leetcode.com/problems/equal-rational-numbers/
Best: Parse to fraction — O(1)
"""

from fractions import Fraction
import re


class Solution:
    def isRationalEqual(self, s: str, t: str) -> bool:
        def to_frac(x: str) -> Fraction:
            m = re.fullmatch(r"(\d+)(?:\.(\d*)(?:\((\d+)\))?)?", x)
            integer, nonrep, rep = m.group(1), m.group(2) or "", m.group(3)
            if rep is None:
                if not nonrep:
                    return Fraction(int(integer), 1)
                return Fraction(int(integer + nonrep), 10 ** len(nonrep))
            # a.b(c) = a.b + 0.00..0(c) with |b| zeros after decimal before repeat
            # value = integer + nonrep/10^|nonrep| + rep / (10^|nonrep| * (10^|rep|-1))
            base = Fraction(int(integer), 1)
            if nonrep:
                base += Fraction(int(nonrep), 10 ** len(nonrep))
            base += Fraction(int(rep), (10 ** len(nonrep)) * (10 ** len(rep) - 1))
            return base

        return to_frac(s) == to_frac(t)


def _test():
    s = Solution()
    assert s.isRationalEqual("0.(52)", "0.5(25)") is True
    assert s.isRationalEqual("0.1666(6)", "0.166(6)") is True
    assert s.isRationalEqual("0.9(9)", "1.") is True
    assert s.isRationalEqual("0.12", "0.12(0)") is True
    print("LC 972 OK")


if __name__ == "__main__":
    _test()
