"""
LC 3154. Find Number of Ways to Reach the K-th Stair — Hard
https://leetcode.com/problems/find-number-of-ways-to-reach-the-k-th-stair/
Best: Enumerate jump count + combinatorics — O(log k)
"""

from math import comb


class Solution:
    def waysToReachStair(self, k: int) -> int:
        # start at 1; j jumps add 2^0+...+2^{j-1}=2^j-1 → position 2^j
        ans = 0
        for j in range(31):
            pos = 1 << j
            downs = pos - k
            if 0 <= downs <= j + 1:
                ans += comb(j + 1, downs)
        return ans


def _test():
    s = Solution()
    assert s.waysToReachStair(0) == 2
    assert s.waysToReachStair(1) == 4
    print("LC 3154 OK")


if __name__ == "__main__":
    _test()
