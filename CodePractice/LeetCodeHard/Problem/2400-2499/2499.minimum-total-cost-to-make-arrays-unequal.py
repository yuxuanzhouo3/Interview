"""
LC 2499. Minimum Total Cost to Make Arrays Unequal — Hard
https://leetcode.com/problems/minimum-total-cost-to-make-arrays-unequal/
Best: Count conflict positions + greedy with mode — O(n)
"""

from typing import List
from collections import Counter


class Solution:
    def minimumTotalCost(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        same = []
        for i in range(n):
            if nums1[i] == nums2[i]:
                same.append(i)
        m = len(same)
        if m == 0:
            return 0
        cnt = Counter(nums1[i] for i in same)
        mode, freq = cnt.most_common(1)[0]
        # need enough partners: if freq > m/2 need extras from outside
        need = max(0, 2 * freq - m)
        cost = sum(same)
        if need == 0:
            return cost
        # collect indices not in same that can help (value != mode on either side swap)
        extras = []
        same_set = set(same)
        for i in range(n):
            if i in same_set:
                continue
            if nums1[i] != mode and nums2[i] != mode:
                extras.append(i)
        if len(extras) < need:
            return -1
        extras.sort()
        return cost + sum(extras[:need])


def _test():
    s = Solution()
    assert s.minimumTotalCost([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 10
    assert s.minimumTotalCost([2, 2, 2, 1, 3], [1, 2, 2, 3, 3]) == 10
    assert s.minimumTotalCost([1, 2, 2], [1, 2, 2]) == -1
    print("LC 2499 OK")


if __name__ == "__main__":
    _test()
