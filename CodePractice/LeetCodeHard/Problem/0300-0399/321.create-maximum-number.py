"""
LC 321. Create Maximum Number — Hard
Best: Pick i from nums1, k-i from nums2; merge max — O(k*(n+m)^2) typical
"""

from typing import List


class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def pick(nums, t):
            drop = len(nums) - t
            st = []
            for x in nums:
                while drop and st and st[-1] < x:
                    st.pop()
                    drop -= 1
                st.append(x)
            return st[:t]

        def merge(a, b):
            return [max(a, b).pop(0) for _ in range(len(a) + len(b))]

        m, n = len(nums1), len(nums2)
        best = []
        for i in range(max(0, k - n), min(k, m) + 1):
            cand = merge(pick(nums1, i), pick(nums2, k - i))
            if cand > best:
                best = cand
        return best


def _test():
    s = Solution()
    assert s.maxNumber([3, 4, 6, 5], [9, 1, 2, 5, 8, 3], 5) == [9, 8, 6, 5, 3]
    assert s.maxNumber([6, 7], [6, 0, 4], 5) == [6, 7, 6, 0, 4]
    print("LC 321 OK")


if __name__ == "__main__":
    _test()
