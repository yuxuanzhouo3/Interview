"""
LC 810. Chalkboard XOR Game — Hard
Best: Math — win iff xor==0 or n even — O(n)
"""

from typing import List
from functools import reduce
import operator


class Solution:
    def xorGame(self, nums: List[int]) -> bool:
        return reduce(operator.xor, nums) == 0 or len(nums) % 2 == 0


def _test():
    s = Solution()
    assert s.xorGame([1, 1, 2]) is False
    assert s.xorGame([0, 1]) is True
    print("LC 810 OK")


if __name__ == "__main__":
    _test()
