"""
LC 84. Largest Rectangle in Histogram — Hard
Best: Monotonic stack — O(n)
"""

from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        st = []  # increasing indices
        ans = 0
        for i, h in enumerate(heights + [0]):
            while st and heights[st[-1]] > h:
                height = heights[st.pop()]
                left = st[-1] if st else -1
                ans = max(ans, height * (i - left - 1))
            st.append(i)
        return ans


def _test():
    s = Solution()
    assert s.largestRectangleArea([2, 1, 5, 6, 2, 3]) == 10
    assert s.largestRectangleArea([2, 4]) == 4
    print("LC 84 OK")


if __name__ == "__main__":
    _test()
