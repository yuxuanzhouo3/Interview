"""
LC 3312. Sorted GCD Pair Queries — Hard
https://leetcode.com/problems/sorted-gcd-pair-queries/
Best: 方法一：预处理 + 前缀和 + 二分查找
"""

from collections import Counter
from bisect import bisect_right
from itertools import accumulate

class Solution:
    def gcdValues(self, nums: list[int], queries: list[int]) -> list[int]:
        mx = max(nums)
        cnt = Counter(nums)
        cnt_g = [0] * (mx + 1)
        for i in range(mx, 0, -1):
            v = 0
            for j in range(i, mx + 1, i):
                v += cnt[j]
                cnt_g[i] -= cnt_g[j]
            cnt_g[i] += v * (v - 1) // 2
        s = list(accumulate(cnt_g))
        return [bisect_right(s, q) for q in queries]

def _test():
    s = Solution()
    assert s.gcdValues(nums=[2, 3, 4], queries=[0, 2, 2]) == [1, 2, 2]
    assert s.gcdValues(nums=[4, 4, 2, 1], queries=[5, 3, 1, 0]) == [4, 2, 1, 1]
    assert s.gcdValues(nums=[2, 2], queries=[0, 0]) == [2, 2]
    print("LC 3312 OK")

if __name__ == "__main__":
    _test()
