"""
LC 2972. Count the Number of Incremovable Subarrays II — Hard
https://leetcode.com/problems/count-the-number-of-incremovable-subarrays-ii/
Best: Two pointers on incremovable — O(n)
"""

from typing import List


class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        # find longest strict increasing prefix and suffix
        i = 0
        while i + 1 < n and nums[i] < nums[i + 1]:
            i += 1
        if i == n - 1:
            return n * (n + 1) // 2
        j = n - 1
        while j - 1 >= 0 and nums[j - 1] < nums[j]:
            j -= 1
        # remove mid; keep prefix[0..a] and suffix[b..n)
        # a in [-1..i], b in [j..n]
        ans = 0
        # empty left: a=-1, any b in [j..n] -> n-j+1, plus removing all? 
        # count: for each left end a=-1..i, find minimal b>=j with (a==-1 or nums[a]<nums[b])
        ans = i + 2  # remove to leave only suffix starting at >=j, or leave nothing? 
        # Standard:
        # Keeping only suffix: b from j to n (n means empty suffix) => n-j+1 ways with empty prefix
        # Keeping only prefix: a from -1 to i ( -1 empty) but wait double count empty
        ans = 0
        b = j
        for a in range(-1, i + 1):
            while b < n and (a >= 0 and nums[a] >= nums[b]):
                b += 1
            ans += n - b + 1  # b..n inclusive empty
        return ans


def _test():
    s = Solution()
    assert s.incremovableSubarrayCount([1, 2, 3, 4]) == 10
    assert s.incremovableSubarrayCount([6, 5, 7, 8]) == 7
    assert s.incremovableSubarrayCount([8, 7, 6, 6]) == 3
    print("LC 2972 OK")


if __name__ == "__main__":
    _test()

