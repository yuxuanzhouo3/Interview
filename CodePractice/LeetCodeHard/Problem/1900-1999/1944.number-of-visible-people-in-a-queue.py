"""
LC 1944. Number of Visible People in a Queue — Hard
https://leetcode.com/problems/number-of-visible-people-in-a-queue/
Best: monotonic decreasing stack from right — O(n)
"""

from typing import List


class Solution:
    def canSeePersonsCount(self, heights: List[int]) -> List[int]:
        n = len(heights)
        ans = [0] * n
        stack = []  # decreasing heights indices from right
        for i in range(n - 1, -1, -1):
            cnt = 0
            while stack and heights[stack[-1]] < heights[i]:
                stack.pop()
                cnt += 1
            if stack:
                cnt += 1  # first taller/equal blocks further
            ans[i] = cnt
            stack.append(i)
        return ans


def _test():
    s = Solution()
    assert s.canSeePersonsCount([10, 6, 8, 5, 11, 9]) == [3, 1, 2, 1, 1, 0]
    assert s.canSeePersonsCount([5, 1, 2, 3, 10]) == [4, 1, 1, 1, 0]
    print("LC 1944 OK")


if __name__ == "__main__":
    _test()
