"""
LC 2449. Minimum Number of Operations to Make Arrays Similar — Hard
https://leetcode.com/problems/minimum-number-of-operations-to-make-arrays-similar/
Best: Sort same parity; sum |diff|/4 — O(n log n)
"""

from typing import List


class Solution:
    def makeSimilar(self, nums: List[int], target: List[int]) -> int:
        def split(a: List[int]):
            odd = sorted(x for x in a if x % 2)
            even = sorted(x for x in a if x % 2 == 0)
            return odd, even

        no, ne = split(nums)
        to, te = split(target)
        ans = 0
        for a, b in zip(no + ne, to + te):
            ans += abs(a - b)
        return ans // 4


def _test():
    s = Solution()
    assert s.makeSimilar([8, 12, 6], [2, 14, 10]) == 2
    assert s.makeSimilar([1, 2, 5], [4, 1, 3]) == 1
    assert s.makeSimilar([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]) == 0
    print("LC 2449 OK")


if __name__ == "__main__":
    _test()
