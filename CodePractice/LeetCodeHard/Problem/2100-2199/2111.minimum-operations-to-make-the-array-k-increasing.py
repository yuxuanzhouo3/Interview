"""
LC 2111. Minimum Operations to Make the Array K-Increasing — Hard
https://leetcode.com/problems/minimum-operations-to-make-the-array-k-increasing/
Best: per residue class LIS (non-decreasing) — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def kIncreasing(self, arr: List[int], k: int) -> int:
        def lis_nondec(a: List[int]) -> int:
            tails: List[int] = []
            for x in a:
                i = bisect.bisect_right(tails, x)
                if i == len(tails):
                    tails.append(x)
                else:
                    tails[i] = x
            return len(tails)

        ans = 0
        n = len(arr)
        for r in range(k):
            seq = [arr[i] for i in range(r, n, k)]
            ans += len(seq) - lis_nondec(seq)
        return ans


def _test():
    s = Solution()
    assert s.kIncreasing([5, 4, 3, 2, 1], 1) == 4
    assert s.kIncreasing([4, 1, 5, 2, 6, 2], 2) == 0
    assert s.kIncreasing([4, 1, 5, 2, 6, 2], 3) == 2
    print("LC 2111 OK")


if __name__ == "__main__":
    _test()
