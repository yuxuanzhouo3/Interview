"""
LC 514. Freedom Trail — Hard
https://leetcode.com/problems/freedom-trail/
Best: DP on ring positions per key char — O(|key| * |ring|^2)
"""

from collections import defaultdict
from functools import lru_cache


class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        pos = defaultdict(list)
        for i, c in enumerate(ring):
            pos[c].append(i)
        n = len(ring)

        @lru_cache(None)
        def dp(k_idx: int, r_idx: int) -> int:
            if k_idx == len(key):
                return 0
            best = float("inf")
            for p in pos[key[k_idx]]:
                dist = abs(p - r_idx)
                dist = min(dist, n - dist)
                best = min(best, dist + 1 + dp(k_idx + 1, p))
            return int(best)

        return dp(0, 0)


def _test():
    s = Solution()
    assert s.findRotateSteps("godding", "gd") == 4
    assert s.findRotateSteps("godding", "godding") == 13
    assert s.findRotateSteps("abc", "a") == 1
    print("LC 514 OK")


if __name__ == "__main__":
    _test()
