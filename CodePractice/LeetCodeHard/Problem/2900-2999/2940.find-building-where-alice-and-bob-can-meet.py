"""
LC 2940. Find Building Where Alice and Bob Can Meet — Hard
https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/
Best: Offline queries + mono decreasing stack from right — O((n+q) log n)
"""

from typing import List


class Solution:
    def leftmostBuildingQueries(self, heights: List[int], queries: List[List[int]]) -> List[int]:
        n = len(heights)
        ans = [-1] * len(queries)
        pending = [[] for _ in range(n)]
        for qi, (a, b) in enumerate(queries):
            if a > b:
                a, b = b, a
            if a == b or heights[a] < heights[b]:
                ans[qi] = b
            else:
                pending[b].append((heights[a], qi))

        stack = []  # indices to the right; heights strictly decreasing toward the end
        for i in range(n - 1, -1, -1):
            for need, qi in pending[i]:
                lo, hi = 0, len(stack) - 1
                best = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if heights[stack[mid]] > need:
                        best = stack[mid]
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best != -1:
                    ans[qi] = best
            while stack and heights[stack[-1]] <= heights[i]:
                stack.pop()
            stack.append(i)
        return ans


def _test():
    s = Solution()
    assert s.leftmostBuildingQueries([6, 4, 8, 5, 2, 7], [[0, 1], [0, 3], [2, 4], [3, 4], [2, 2]]) == [2, 5, -1, 5, 2]
    assert s.leftmostBuildingQueries([5, 3, 8, 2, 6, 1, 4, 6], [[0, 7], [3, 5], [5, 2], [3, 0], [1, 6]]) == [7, 6, -1, 4, 6]
    print("LC 2940 OK")


if __name__ == "__main__":
    _test()
