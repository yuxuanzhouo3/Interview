"""
LC 3036. Number of Subarrays That Match a Pattern II — Hard
https://leetcode.com/problems/number-of-subarrays-that-match-a-pattern-ii/
Best: KMP on mapped array — O(n+m)
"""

from typing import List


class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        m = len(pattern)
        n = len(nums)
        # map nums diffs to -1/0/1 sequence length n-1
        a = []
        for i in range(n - 1):
            if nums[i + 1] > nums[i]:
                a.append(1)
            elif nums[i + 1] == nums[i]:
                a.append(0)
            else:
                a.append(-1)
        # KMP count pattern in a
        p = pattern
        # build lps
        lps = [0] * m
        length = 0
        i = 1
        while i < m:
            if p[i] == p[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        ans = i = j = 0
        while i < len(a):
            if a[i] == p[j]:
                i += 1
                j += 1
                if j == m:
                    ans += 1
                    j = lps[j - 1]
            elif j:
                j = lps[j - 1]
            else:
                i += 1
        return ans


def _test():
    s = Solution()
    assert s.countMatchingSubarrays([1, 2, 3, 4, 5, 6], [1, 1]) == 4
    assert s.countMatchingSubarrays([1, 4, 4, 1, 3, 5, 5, 3], [1, 0, -1]) == 2
    print("LC 3036 OK")


if __name__ == "__main__":
    _test()

