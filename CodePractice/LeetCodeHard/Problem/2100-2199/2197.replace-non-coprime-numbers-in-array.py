"""
LC 2197. Replace Non-Coprime Numbers in Array — Hard
https://leetcode.com/problems/replace-non-coprime-numbers-in-array/
Best: stack + gcd/lcm merge — O(n log A)
"""

from typing import List
from math import gcd


class Solution:
    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        st = []
        for x in nums:
            while st:
                g = gcd(st[-1], x)
                if g == 1:
                    break
                x = st.pop() // g * x  # lcm
            st.append(x)
        return st


def _test():
    s = Solution()
    assert s.replaceNonCoprimes([6, 4, 3, 2, 7, 6, 2]) == [12, 7, 6]
    assert s.replaceNonCoprimes([2, 2, 1, 1, 3, 3, 3]) == [2, 1, 1, 3]
    print("LC 2197 OK")


if __name__ == "__main__":
    _test()
