"""
LC 1923. Longest Common Subpath — Hard
https://leetcode.com/problems/longest-common-subpath/
Best: Binary search + rolling hash — O(L log L)
"""

from collections import Counter
from typing import List

class Solution:
    def longestCommonSubpath(self, n: int, paths: List[List[int]]) -> int:
        def check(k: int) -> bool:
            if k == 0:
                return True
            cnt = Counter()
            for h in hh:
                vis = set()
                for i in range(1, len(h) - k + 1):
                    j = i + k - 1
                    x = (h[j] - h[i - 1] * p[j - i + 1]) % mod
                    if x not in vis:
                        vis.add(x)
                        cnt[x] += 1
            return bool(cnt) and max(cnt.values()) == m

        m = len(paths)
        mx = max(len(path) for path in paths)
        base = 133331
        mod = 2**64 + 1
        p = [1] * (mx + 1)
        for i in range(1, len(p)):
            p[i] = p[i - 1] * base % mod
        hh = []
        for path in paths:
            k = len(path)
            h = [0] * (k + 1)
            for i, x in enumerate(path, 1):
                h[i] = (h[i - 1] * base + x) % mod
            hh.append(h)
        lo, hi = 0, min(len(path) for path in paths)
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if check(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

def _test():

    s = Solution()
    assert s.longestCommonSubpath(5, [[0, 1, 2, 3, 4], [2, 3, 4], [4, 0, 1, 2, 3]]) == 2
    assert s.longestCommonSubpath(3, [[0], [1], [2]]) == 0
    assert s.longestCommonSubpath(5, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]) == 1
    print("LC 1923 OK")

if __name__ == "__main__":
    _test()
