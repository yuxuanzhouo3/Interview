"""
LC 2945. Find Maximum Non-decreasing Array Length — Hard
https://leetcode.com/problems/find-maximum-non-decreasing-array-length/
Best: Prefix + monoqueue binary search — O(n log n)
"""

from typing import List


class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        pre = [0] * (n + 1)
        for i, x in enumerate(nums):
            pre[i + 1] = pre[i] + x
        s = [0] * (n + 1)
        dp = [0] * (n + 1)
        q = [0]
        for i in range(1, n + 1):
            lo, hi = 0, len(q) - 1
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if s[q[mid]] <= pre[i]:
                    best = q[mid]
                    lo = mid + 1
                else:
                    hi = mid - 1
            dp[i] = dp[best] + 1
            last = pre[i] - pre[best]
            s[i] = pre[i] + last
            while q and s[q[-1]] >= s[i]:
                q.pop()
            q.append(i)
        return dp[n]


def _test():
    s = Solution()
    assert s.findMaximumLength([5, 2, 2]) == 1
    assert s.findMaximumLength([1, 2, 3, 4]) == 4
    assert s.findMaximumLength([4, 3, 2, 6]) == 3
    print("LC 2945 OK")


if __name__ == "__main__":
    _test()
