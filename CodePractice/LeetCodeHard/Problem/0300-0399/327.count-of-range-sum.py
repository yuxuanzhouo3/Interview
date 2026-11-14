"""
LC 327. Count of Range Sum — Hard
Best: Merge sort on prefix sums — O(n log n)
"""

from typing import List


class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        pref = [0]
        for x in nums:
            pref.append(pref[-1] + x)
        self.ans = 0

        def sort(arr):
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = sort(arr[:mid])
            right = sort(arr[mid:])
            j = k = 0
            for x in left:
                while j < len(right) and right[j] - x < lower:
                    j += 1
                while k < len(right) and right[k] - x <= upper:
                    k += 1
                self.ans += k - j
            merged = []
            i = r = 0
            while i < len(left) or r < len(right):
                if r == len(right) or (i < len(left) and left[i] <= right[r]):
                    merged.append(left[i]); i += 1
                else:
                    merged.append(right[r]); r += 1
            return merged

        sort(pref)
        return self.ans


def _test():
    s = Solution()
    assert s.countRangeSum([-2, 5, -1], -2, 2) == 3
    assert s.countRangeSum([0], 0, 0) == 1
    print("LC 327 OK")


if __name__ == "__main__":
    _test()
