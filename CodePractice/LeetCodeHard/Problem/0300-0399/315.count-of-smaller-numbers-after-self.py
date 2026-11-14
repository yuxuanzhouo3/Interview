"""
LC 315. Count of Smaller Numbers After Self — Hard
Best: Merge sort count / Fenwick on ranks — O(n log n)
"""

from typing import List


class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n
        # merge sort indices
        idx = list(range(n))

        def merge_sort(a):
            if len(a) <= 1:
                return a
            mid = len(a) // 2
            left = merge_sort(a[:mid])
            right = merge_sort(a[mid:])
            merged = []
            i = j = 0
            while i < len(left) or j < len(right):
                if j == len(right) or (i < len(left) and nums[left[i]] <= nums[right[j]]):
                    ans[left[i]] += j
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
            return merged

        merge_sort(idx)
        return ans


def _test():
    s = Solution()
    assert s.countSmaller([5, 2, 6, 1]) == [2, 1, 1, 0]
    assert s.countSmaller([-1]) == [0]
    print("LC 315 OK")


if __name__ == "__main__":
    _test()
