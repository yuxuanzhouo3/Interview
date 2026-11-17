"""
LC 3830. Longest Alternating Subarray After Removing At Most One Element — Hard
https://leetcode.com/problems/longest-alternating-subarray-after-removing-at-most-one-element/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def longestAlternating(self, nums: list[int]) -> int:
        n = len(nums)
        l1 = [1] * n
        l2 = [1] * n
        r1 = [1] * n
        r2 = [1] * n
        ans = 0
        for i in range(1, n):
            if nums[i - 1] < nums[i]:
                l1[i] = l2[i - 1] + 1
            elif nums[i - 1] > nums[i]:
                l2[i] = l1[i - 1] + 1
            ans = max(ans, l1[i], l2[i])
        for i in range(n - 2, -1, -1):
            if nums[i + 1] > nums[i]:
                r1[i] = r2[i + 1] + 1
            elif nums[i + 1] < nums[i]:
                r2[i] = r1[i + 1] + 1
        for i in range(1, n - 1):
            if nums[i - 1] < nums[i + 1]:
                ans = max(ans, l2[i - 1] + r2[i + 1])
            elif nums[i - 1] > nums[i + 1]:
                ans = max(ans, l1[i - 1] + r1[i + 1])
        return ans

def _test():
    s = Solution()
    assert s.longestAlternating([2,1,3,2]) == 4
    assert s.longestAlternating([3,2,1,2,3,2,1]) == 4
    assert s.longestAlternating([100000,100000]) == 1
    print("LC 3830 OK")


if __name__ == "__main__":
    _test()
