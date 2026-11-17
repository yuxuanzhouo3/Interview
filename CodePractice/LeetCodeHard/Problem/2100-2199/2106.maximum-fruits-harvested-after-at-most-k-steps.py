"""
LC 2106. Maximum Fruits Harvested After at Most K Steps — Hard
https://leetcode.com/problems/maximum-fruits-harvested-after-at-most-k-steps/
Best: Prefix + two-direction windows — O(n)
"""


from typing import List
import bisect


class Solution:
    def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
        n = len(fruits)
        pos = [f[0] for f in fruits]
        pref = [0] * (n + 1)
        for i in range(n):
            pref[i + 1] = pref[i] + fruits[i][1]

        def sum_range(L, R):
            if L > R:
                return 0
            return pref[R + 1] - pref[L]

        ans = 0
        # go left then right, or right then left
        # window [left, right] reachable within k steps from startPos
        j = 0
        for i in range(n):
            # expand right endpoint i, move left j
            while j <= i and min(
                abs(startPos - pos[j]) + (pos[i] - pos[j]),
                abs(startPos - pos[i]) + (pos[i] - pos[j]),
            ) > k:
                j += 1
            if j <= i:
                ans = max(ans, sum_range(j, i))
        return ans


def _test():
    s = Solution()
    assert s.maxTotalFruits([[2, 8], [6, 3], [8, 6]], 5, 4) == 9
    assert s.maxTotalFruits([[0, 9], [4, 1], [5, 7], [6, 2], [7, 4], [10, 9]], 5, 4) == 14
    assert s.maxTotalFruits([[0, 3], [6, 4], [8, 5]], 3, 2) == 0
    print("LC 2106 OK")


if __name__ == "__main__":
    _test()

