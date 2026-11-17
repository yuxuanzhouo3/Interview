"""
LC 4105. Repeated String Match II — Hard
https://leetcode.com/problems/repeated-string-match-ii/
Best: check if b*y is substring of a*x without building huge strings (period / KMP on bounded copies)
"""


class Solution:
    def repeatedStringMatch(self, a: str, b: str, x: int, y: int) -> bool:
        # Need |a|*x >= |b|*y
        la, lb = len(a), len(b)
        if la * x < lb * y:
            return False
        # b*y must appear in a*x. Since strings are short (<=100) but x,y huge,
        # reduce: b must be composed from a's rotation / a itself.
        # Equivalent: (b * y) in (a * x).
        # Let need = b * y length. We only need to check if some alignment works.
        # Build enough of a to cover any substring of length lb*y that fits in a*x.
        # Max needed copies of a to test substring of length L is at most (L/la)+2,
        # but capped by x.
        need_len = lb * y
        # If y is huge, b*y is periodic with period dividing. Check via:
        # exists start s in [0, la*x - need_len] s.t. for all i, (a*(...))[s+i] == (b*y)[i]
        # Which means a[(s+i)%la] == b[i%lb] for all i < need_len, and the covered
        # range of a-index must lie within x copies: s+need_len-1 < la*x.

        # For all i < need_len: a[(s+i) % la] == b[i % lb]
        # So for each residue, consistency between a and b cycles.
        # Try all s mod lcm(la,lb) but s in 0..la-1 is enough for alignment of a,
        # then verify the infinite period match for need_len and that end < la*x.

        for s in range(la):
            # end index exclusive = s + need_len
            if s + need_len > la * x:
                break
            ok = True
            # Check one full period of combined constraints: length lcm or la*lb
            # Sufficient to check for i in 0..la*lb-1 and also that pattern extends
            # Actually check i=0..max(la,lb)*2 or all positions via cycle consistency:
            for i in range(la * lb):
                if i >= need_len:
                    break
                if a[(s + i) % la] != b[i % lb]:
                    ok = False
                    break
            if not ok:
                continue
            # Pattern matches for la*lb prefix of infinite; since both periodic,
            # it matches for all i if it matches for i < lcm(la,lb) <= la*lb.
            return True
        return False


def _test():
    s = Solution()
    assert s.repeatedStringMatch("ab", "abab", 2, 1) is True
    assert s.repeatedStringMatch("abc", "bc", 2, 3) is False
    assert s.repeatedStringMatch("abcd", "cdab", 2, 1) is True
    print("LC 4105 OK")


if __name__ == "__main__":
    _test()
