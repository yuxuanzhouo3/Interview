"""
LC 1889. Minimum Space Wasted From Packaging — Hard
https://leetcode.com/problems/minimum-space-wasted-from-packaging/
Best: sort + prefix + binary search — O(n log n + m log m)
"""

from typing import List
import bisect


class Solution:
    def minWastedSpace(self, packages: List[int], boxes: List[List[int]]) -> int:
        MOD = 10**9 + 7
        packages.sort()
        n = len(packages)
        pref = [0]
        for p in packages:
            pref.append(pref[-1] + p)
        total = pref[-1]
        ans = float("inf")
        for box in boxes:
            box = sorted(box)
            if box[-1] < packages[-1]:
                continue
            prev = 0
            waste = 0
            used = 0
            for b in box:
                idx = bisect.bisect_right(packages, b)
                # packages[prev:idx] go into boxes of size b
                cnt = idx - prev
                waste += b * cnt - (pref[idx] - pref[prev])
                prev = idx
                if prev == n:
                    break
            ans = min(ans, waste)
        return -1 if ans == float("inf") else ans % MOD


def _test():
    s = Solution()
    assert s.minWastedSpace([2, 3, 5], [[4, 8], [2, 8]]) == 6
    assert s.minWastedSpace([2, 3, 5], [[1, 4], [2, 3], [3, 4]]) == -1
    assert s.minWastedSpace([3, 5, 8, 10, 11, 12], [[12], [11, 9], [10, 5, 14]]) == 9
    print("LC 1889 OK")


if __name__ == "__main__":
    _test()
