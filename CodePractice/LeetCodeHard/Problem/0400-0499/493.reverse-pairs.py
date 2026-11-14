"""
LC 493. Reverse Pairs — Hard
https://leetcode.com/problems/reverse-pairs/
Best: Merge sort count — O(n log n)
"""

from typing import List


class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def merge_count(arr: List[int]) -> int:
            if len(arr) <= 1:
                return 0
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]
            cnt = merge_count(left) + merge_count(right)
            # count pairs across
            j = 0
            for i in range(len(left)):
                while j < len(right) and left[i] > 2 * right[j]:
                    j += 1
                cnt += j
            # merge
            i = j = 0
            k = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
            return cnt

        return merge_count(list(nums))


def _test():
    s = Solution()
    assert s.reversePairs([1, 3, 2, 3, 1]) == 2
    assert s.reversePairs([2, 4, 3, 5, 1]) == 3
    assert s.reversePairs([]) == 0
    assert s.reversePairs([1]) == 0
    print("LC 493 OK")


if __name__ == "__main__":
    _test()
