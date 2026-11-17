"""
LC 2334. Subarray With Elements Greater Than Varying Threshold — Hard
https://leetcode.com/problems/subarray-with-elements-greater-than-varying-threshold/
Best: Monotonic stack max width where min > threshold/len — O(n)
"""


from typing import List


class Solution:
    def validSubarraySize(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        left = [-1] * n
        right = [n] * n
        st = []
        for i, x in enumerate(nums):
            while st and nums[st[-1]] >= x:
                st.pop()
            left[i] = st[-1] if st else -1
            st.append(i)
        st = []
        for i in range(n - 1, -1, -1):
            while st and nums[st[-1]] >= nums[i]:
                st.pop()
            right[i] = st[-1] if st else n
            st.append(i)
        for i, x in enumerate(nums):
            k = right[i] - left[i] - 1
            if x > threshold // k:
                return k
        return -1


def _test():
    s = Solution()
    assert s.validSubarraySize([1, 3, 4, 3, 1], 6) == 3
    assert s.validSubarraySize([6, 5, 6, 5, 8], 7) in (1, 2, 3, 4, 5)
    print("LC 2334 OK")


if __name__ == "__main__":
    _test()

