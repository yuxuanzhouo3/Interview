"""
LC 1815. Maximum Number of Groups Getting Fresh Donuts — Hard
https://leetcode.com/problems/maximum-number-of-groups-getting-fresh-donuts/
Best: 余数贪心配对 + 状态压缩记忆化 — O(batchSize^groups)
"""

from typing import List
from functools import lru_cache


class Solution:
    def maxHappyGroups(self, batchSize: int, groups: List[int]) -> int:
        freq = [0] * batchSize
        happy = 0
        for g in groups:
            r = g % batchSize
            if r == 0:
                happy += 1
            elif freq[batchSize - r]:
                freq[batchSize - r] -= 1
                happy += 1
            else:
                freq[r] += 1

        @lru_cache(None)
        def dp(state, rem):
            state = list(state)
            ans = 0
            if any(state):
                for i, f in enumerate(state):
                    if f:
                        state[i] -= 1
                        ans = max(ans, dp(tuple(state), (rem + i) % batchSize))
                        state[i] += 1
                if rem == 0:
                    ans += 1
            return ans

        return happy + dp(tuple(freq), 0)


def _test():
    s = Solution()
    assert s.maxHappyGroups(3, [1, 2, 3, 4, 5, 6]) == 4
    assert s.maxHappyGroups(4, [1, 3, 2, 5, 2, 2, 1, 6]) == 4
    print("LC 1815 OK")


if __name__ == "__main__":
    _test()
