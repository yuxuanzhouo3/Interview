"""
LC 3145. Find Products of Elements of Big Array — Hard
https://leetcode.com/problems/find-products-of-elements-of-big-array/
Best: Bit count prefix + pow2 — O(q * 50)
"""

from typing import List

_M = 50
_cnt = [0] * (_M + 1)
_s = [0] * (_M + 1)
_p = 1
for _i in range(1, _M + 1):
    _cnt[_i] = _cnt[_i - 1] * 2 + _p
    _s[_i] = _s[_i - 1] * 2 + _p * (_i - 1)
    _p *= 2


def _num_idx_and_sum(x: int):
    idx = total = 0
    while x:
        i = x.bit_length() - 1
        idx += _cnt[i]
        total += _s[i]
        x -= 1 << i
        total += (x + 1) * i
        idx += x + 1
    return idx, total


def _f(i: int) -> int:
    l, r = 0, 1 << _M
    while l < r:
        mid = (l + r + 1) >> 1
        idx, _ = _num_idx_and_sum(mid)
        if idx < i:
            l = mid
        else:
            r = mid - 1
    idx, total = _num_idx_and_sum(l)
    i -= idx
    x = l + 1
    for _ in range(i):
        y = x & -x
        total += y.bit_length() - 1
        x -= y
    return total


class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        return [pow(2, _f(right + 1) - _f(left), mod) for left, right, mod in queries]


def _test():
    s = Solution()
    assert s.findProductsOfElements([[1, 3, 7]]) == [4]
    assert s.findProductsOfElements([[2, 5, 3], [7, 7, 4]]) == [2, 2]
    print("LC 3145 OK")


if __name__ == "__main__":
    _test()
