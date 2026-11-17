"""
LC 2736. Maximum Sum Queries — Hard
https://leetcode.com/problems/maximum-sum-queries/
Best: Sort + mono stack + binary — O((n+q) log n)
"""

from typing import List
import bisect


class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        pts = sorted(zip(nums1, nums2), reverse=True)  # by nums1 desc
        qs = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], reverse=True)
        ans = [-1] * len(queries)
        # stack of (nums2, nums1+nums2) increasing nums2, decreasing sum
        stack = []  # pairs (b, a+b) with b increasing
        j = 0
        for x, y, qi in qs:
            while j < n and pts[j][0] >= x:
                a, b = pts[j]
                s = a + b
                while stack and stack[-1][1] <= s:
                    stack.pop()
                if not stack or stack[-1][0] < b:
                    stack.append((b, s))
                j += 1
            # find first in stack with b >= y, max s (stack s decreasing so first ok)
            # binary search
            lo, hi = 0, len(stack)
            while lo < hi:
                mid = (lo + hi) // 2
                if stack[mid][0] >= y:
                    hi = mid
                else:
                    lo = mid + 1
            if lo < len(stack):
                ans[qi] = stack[lo][1]
        return ans


def _test():
    s = Solution()
    assert s.maximumSumQueries([4, 3, 1, 2], [2, 4, 9, 5], [[4, 1], [1, 3], [2, 5]]) == [6, 10, 7]
    assert s.maximumSumQueries([3, 2, 5], [2, 3, 4], [[4, 4], [3, 2], [1, 1]]) == [9, 9, 9]
    print("LC 2736 OK")


if __name__ == "__main__":
    _test()

