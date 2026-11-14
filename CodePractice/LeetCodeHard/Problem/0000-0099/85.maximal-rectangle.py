"""
LC 85. Maximal Rectangle — Hard
Best: Per-row histogram + LC84 stack — O(rows*cols)
"""

from typing import List


class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        cols = len(matrix[0])
        heights = [0] * cols
        ans = 0

        def largest(h):
            st, best = [], 0
            for i, x in enumerate(h + [0]):
                while st and h[st[-1]] > x:
                    height = h[st.pop()]
                    left = st[-1] if st else -1
                    best = max(best, height * (i - left - 1))
                st.append(i)
            return best

        for row in matrix:
            for j, v in enumerate(row):
                heights[j] = heights[j] + 1 if v == "1" else 0
            ans = max(ans, largest(heights))
        return ans


def _test():
    s = Solution()
    m = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
    assert s.maximalRectangle(m) == 6
    print("LC 85 OK")


if __name__ == "__main__":
    _test()
