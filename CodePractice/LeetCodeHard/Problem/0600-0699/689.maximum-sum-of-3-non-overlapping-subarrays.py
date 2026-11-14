"""
LC 689. Maximum Sum of 3 Non-Overlapping Subarrays — Hard
https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/
Best: prefix window sums + left/right best — O(n)
"""

from typing import List


class Solution:
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        w = [0] * (n - k + 1)
        s = sum(nums[:k])
        w[0] = s
        for i in range(1, len(w)):
            s += nums[i + k - 1] - nums[i - 1]
            w[i] = s

        left = [0] * len(w)
        best = 0
        for i in range(len(w)):
            if w[i] > w[best]:
                best = i
            left[i] = best

        right = [0] * len(w)
        best = len(w) - 1
        for i in range(len(w) - 1, -1, -1):
            if w[i] >= w[best]:
                best = i
            right[i] = best

        ans = None
        max_sum = -1
        for j in range(k, len(w) - k):
            i, r = left[j - k], right[j + k]
            total = w[i] + w[j] + w[r]
            if total > max_sum:
                max_sum = total
                ans = [i, j, r]
        return ans


def _test():
    s = Solution()
    assert s.maxSumOfThreeSubarrays([1, 2, 1, 2, 6, 7, 5, 1], 2) == [0, 3, 5]
    assert s.maxSumOfThreeSubarrays([1, 2, 1, 2, 1, 2, 1, 2, 1], 2) == [0, 2, 4]
    print("LC 689 OK")


if __name__ == "__main__":
    _test()
