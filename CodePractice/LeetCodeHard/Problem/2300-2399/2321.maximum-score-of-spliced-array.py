"""
LC 2321. Maximum Score Of Spliced Array — Hard
https://leetcode.com/problems/maximum-score-of-spliced-array/
Best: Kadane on diff arrays — O(n)
"""

from typing import List


class Solution:
    def maximumsSplicedArray(self, nums1: List[int], nums2: List[int]) -> int:
        def kadane(diff: List[int]) -> int:
            best = cur = 0
            for x in diff:
                cur = max(0, cur + x)
                best = max(best, cur)
            return best

        s1, s2 = sum(nums1), sum(nums2)
        d12 = [b - a for a, b in zip(nums1, nums2)]
        d21 = [a - b for a, b in zip(nums1, nums2)]
        return max(s1 + kadane(d12), s2 + kadane(d21))


def _test():
    s = Solution()
    assert s.maximumsSplicedArray([60, 60, 60], [10, 90, 10]) == 210
    assert s.maximumsSplicedArray([20, 40, 20, 70, 30], [50, 20, 50, 40, 20]) == 220
    assert s.maximumsSplicedArray([7, 11, 13], [1, 1, 1]) == 31
    print("LC 2321 OK")


if __name__ == "__main__":
    _test()
