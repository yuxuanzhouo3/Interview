"""
LC 2426. Number of Pairs Satisfying Inequality — Hard
https://leetcode.com/problems/number-of-pairs-satisfying-inequality/
Best: Merge-sort / BIT count nums1[i]-nums2[i] ≤ nums1[j]-nums2[j]+diff — O(n log n)
"""


from typing import List
import bisect


class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], diff: int) -> int:
        arr = [a - b for a, b in zip(nums1, nums2)]
        # count i < j with arr[i] <= arr[j] + diff
        ans = 0

        def sort_count(a):
            nonlocal ans
            if len(a) <= 1:
                return a
            mid = len(a) // 2
            left = sort_count(a[:mid])
            right = sort_count(a[mid:])
            # for each right x, count left <= x+diff
            j = 0
            for x in right:
                while j < len(left) and left[j] <= x + diff:
                    j += 1
                ans += j
            # merge
            merged = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i]); i += 1
                else:
                    merged.append(right[j]); j += 1
            merged.extend(left[i:]); merged.extend(right[j:])
            return merged

        sort_count(arr)
        return ans


def _test():
    s = Solution()
    assert s.numberOfPairs([3, 2, 5], [2, 2, 1], 1) == 3
    assert s.numberOfPairs([3, -1], [-2, 2], -1) == 0
    print("LC 2426 OK")


if __name__ == "__main__":
    _test()

