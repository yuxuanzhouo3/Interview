"""
LC 330. Patching Array — Hard
Best: Greedy cover [1, miss) — O(n + log n)
"""

from typing import List


class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        miss = 1
        patches = i = 0
        while miss <= n:
            if i < len(nums) and nums[i] <= miss:
                miss += nums[i]
                i += 1
            else:
                miss += miss
                patches += 1
        return patches


def _test():
    s = Solution()
    assert s.minPatches([1, 3], 6) == 1
    assert s.minPatches([1, 5, 10], 20) == 2
    print("LC 330 OK")


if __name__ == "__main__":
    _test()
