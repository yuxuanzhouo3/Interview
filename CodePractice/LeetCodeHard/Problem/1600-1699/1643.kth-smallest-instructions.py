"""
LC 1643. Kth Smallest Instructions — Hard
https://leetcode.com/problems/kth-smallest-instructions/
Best: combinatorial choose H vs V — O((v+h)^2) or O(v+h) with pre C
"""

from typing import List
from math import comb


class Solution:
    def kthSmallestPath(self, destination: List[int], k: int) -> str:
        v, h = destination
        ans = []
        while v + h > 0:
            if h > 0:
                # paths starting with H: C(v+h-1, v)
                ways = comb(v + h - 1, v)
                if ways >= k:
                    ans.append("H")
                    h -= 1
                else:
                    ans.append("V")
                    k -= ways
                    v -= 1
            else:
                ans.append("V")
                v -= 1
        return "".join(ans)


def _test():
    s = Solution()
    assert s.kthSmallestPath([2, 3], 1) == "HHHVV"
    assert s.kthSmallestPath([2, 3], 2) == "HHVHV"
    assert s.kthSmallestPath([2, 3], 3) == "HHVVH"
    print("LC 1643 OK")


if __name__ == "__main__":
    _test()
