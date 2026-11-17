"""
LC 3177. Find the Maximum Length of a Good Subsequence II — Hard
https://leetcode.com/problems/find-the-maximum-length-of-a-good-subsequence-ii/
Best: DP map by value & changes — O(nk)
"""

from typing import List
from collections import defaultdict


class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        mp = [defaultdict(int) for _ in range(k + 1)]
        mx = [0] * (k + 1)
        for x in nums:
            upd = []
            for j in range(k + 1):
                cand = mp[j][x] + 1
                if j == 0:
                    cand = max(cand, 1)
                else:
                    cand = max(cand, mx[j - 1] + 1)
                upd.append((j, cand))
            for j, cand in upd:
                if cand > mp[j][x]:
                    mp[j][x] = cand
                mx[j] = max(mx[j], mp[j][x])
        return max(mx)


def _test():
    s = Solution()
    assert s.maximumLength([1, 2, 1, 1, 3], 2) == 4
    assert s.maximumLength([1, 2, 3, 4, 5, 1], 0) == 2
    assert s.maximumLength([1, 2, 3, 4], 1) == 2
    print("LC 3177 OK")


if __name__ == "__main__":
    _test()
