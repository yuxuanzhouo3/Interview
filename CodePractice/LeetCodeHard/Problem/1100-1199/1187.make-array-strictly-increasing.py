"""
LC 1187. Make Array Strictly Increasing — Hard
https://leetcode.com/problems/make-array-strictly-increasing/
Best: DP + binary search on sorted unique arr2 — O(n·m)
"""

import bisect
from typing import List


class Solution:
    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        arr2 = sorted(set(arr2))
        # dp[ops] = min last value with `ops` replacements after processing current index
        INF = 10**18
        dp = {-1: 0}  # last value -> min ops  (use last as key)
        # Better: keep map last -> min cost
        # Process sequentially
        cur = {0: -1}  # ops -> min ending value? Actually track min ops for each last.
        # Standard: dp[j] = min ops using first i of arr1 ending with value...
        # Use: for each position, map from last_value to min operations
        state = {-1: 0}  # last_val -> min_ops

        for a in arr1:
            nxt = {}
            for last, ops in state.items():
                if a > last:
                    nxt[a] = min(nxt.get(a, INF), ops)
                # replace a with smallest arr2 > last
                i = bisect.bisect_right(arr2, last)
                if i < len(arr2):
                    v = arr2[i]
                    nxt[v] = min(nxt.get(v, INF), ops + 1)
            if not nxt:
                return -1
            state = nxt
        return min(state.values())


def _test():
    s = Solution()
    assert s.makeArrayIncreasing([1, 5, 3, 6, 7], [1, 3, 2, 4]) == 1
    assert s.makeArrayIncreasing([1, 5, 3, 6, 7], [4, 3, 1]) == 2
    assert s.makeArrayIncreasing([1, 5, 3, 6, 7], [1, 6, 3, 3]) == -1
    print("LC 1187 OK")


if __name__ == "__main__":
    _test()
