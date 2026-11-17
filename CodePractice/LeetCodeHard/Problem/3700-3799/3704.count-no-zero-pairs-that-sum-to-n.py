"""
LC 3704. Count No-Zero Pairs That Sum to N — Hard
https://leetcode.com/problems/count-no-zero-pairs-that-sum-to-n/
Best: Digit DP (low to high) — O(log n)
"""

from functools import lru_cache


class Solution:
    def countNoZeroPairs(self, n: int) -> int:
        digs = list(map(int, str(n)))[::-1]
        L = len(digs)

        @lru_cache(None)
        def g(pos: int, carry: int, alive_a: bool, alive_b: bool) -> int:
            if pos == L:
                return int(carry == 0 and (not alive_a) and (not alive_b))
            res = 0
            nd = digs[pos]
            choices_a = range(1, 10) if alive_a else (0,)
            choices_b = range(1, 10) if alive_b else (0,)
            for da in choices_a:
                for db in choices_b:
                    total = da + db + carry
                    if total % 10 != nd:
                        continue
                    ncarry = total // 10
                    next_a = (False,) if not alive_a else (True, False)
                    next_b = (False,) if not alive_b else (True, False)
                    for na in next_a:
                        for nb in next_b:
                            res += g(pos + 1, ncarry, na, nb)
            return res

        return g(0, 0, True, True)


def _test():
    s = Solution()
    assert s.countNoZeroPairs(2) == 1
    assert s.countNoZeroPairs(3) == 2
    assert s.countNoZeroPairs(11) == 8
    print("LC 3704 OK")


if __name__ == "__main__":
    _test()
