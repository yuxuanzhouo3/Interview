"""
LC 798. Smallest Rotation with Highest Score — Hard
Best: Diff of score changes — O(n)
"""

from typing import List


class Solution:
    def bestRotation(self, nums: List[int]) -> int:
        n = len(nums)
        change = [1] * n
        for i, a in enumerate(nums):
            # when k = (i - a + 1) % n, this index starts to lose the +1
            change[(i - a + 1) % n] -= 1
        for i in range(1, n):
            change[i] += change[i - 1]
        return change.index(max(change))


def _test():
    s = Solution()
    assert s.bestRotation([2, 3, 1, 4, 0]) == 3
    assert s.bestRotation([1, 3, 0, 2, 4]) == 0
    print("LC 798 OK")


if __name__ == "__main__":
    _test()
