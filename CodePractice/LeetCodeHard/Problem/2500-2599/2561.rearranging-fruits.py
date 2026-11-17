"""
LC 2561. Rearranging Fruits — Hard
https://leetcode.com/problems/rearranging-fruits/
Best: Diff multiset + swap with global min — O(n log n)
"""

from typing import List
from collections import Counter


class Solution:
    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        cnt = Counter(basket1)
        for x in basket2:
            cnt[x] -= 1
        extra = []
        for v, c in cnt.items():
            if c % 2:
                return -1
            for _ in range(abs(c) // 2):
                extra.append(v)
        if not extra:
            return 0
        extra.sort()
        # half of extras need to be swapped (the ones from the heavier side)
        m = len(extra) // 2
        global_min = min(min(basket1), min(basket2))
        ans = 0
        for i in range(m):
            ans += min(extra[i], 2 * global_min)
        return ans


def _test():
    s = Solution()
    assert s.minCost([4, 2, 2, 2], [1, 4, 1, 2]) == 1
    assert s.minCost([2, 3, 4, 1], [3, 2, 5, 1]) == -1
    print("LC 2561 OK")


if __name__ == "__main__":
    _test()
