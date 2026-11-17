"""
LC 2941. Maximum GCD-Sum of a Subarray — Hard
Best: Sparse sparse table / stack of gcd segments — O(n log A)
"""

from typing import List
import math


class Solution:
    def maxGcdSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pref = [0]
        for x in nums:
            pref.append(pref[-1] + x)
        ans = 0
        # segs: list of (left_index, gcd) for segments ending at i
        segs = []
        for i, x in enumerate(nums):
            nsegs = [(i, x)]
            for l, g in segs:
                ng = math.gcd(g, x)
                if nsegs[-1][1] == ng:
                    nsegs[-1] = (l, ng)
                else:
                    nsegs.append((l, ng))
            segs = nsegs
            for l, g in segs:
                if i - l + 1 >= k:
                    ans = max(ans, g * (pref[i + 1] - pref[l]))
        return ans


def _test():
    s = Solution()
    assert s.maxGcdSum([2, 1, 4, 4, 4, 2], 2) == 48
    print("LC 2941 OK")


if __name__ == "__main__":
    _test()
