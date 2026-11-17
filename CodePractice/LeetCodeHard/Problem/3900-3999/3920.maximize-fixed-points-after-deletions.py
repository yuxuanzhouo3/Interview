"""
LC 3920. Maximize Fixed Points After Deletions — Hard
https://leetcode.com/problems/maximize-fixed-points-after-deletions/
Best: sort by (i-x, x) then LIS on values x
"""
import bisect


class Solution:
    def maxFixedPoints(self, nums: list[int]) -> int:
        cands = sorted((i - x, x) for i, x in enumerate(nums) if i >= x)
        tails: list[int] = []
        for _, x in cands:
            pos = bisect.bisect_left(tails, x)
            if pos == len(tails):
                tails.append(x)
            else:
                tails[pos] = x
        return len(tails)


def _test():
    s = Solution()
    assert s.maxFixedPoints([0, 2, 1]) == 2
    assert s.maxFixedPoints([3, 1, 2]) == 2
    assert s.maxFixedPoints([1, 0, 1, 2]) == 3
    print("LC 3920 OK")


if __name__ == "__main__":
    _test()
