"""
LC 3027. Find the Number of Ways to Place People II — Hard
https://leetcode.com/problems/find-the-number-of-ways-to-place-people-ii/
Best: Sort + scan upper-left/lower-right pairs — O(n^2)
"""

from typing import List


class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        points.sort(key=lambda p: (p[0], -p[1]))
        n = len(points)
        ans = 0
        for i in range(n):
            max_y = float("-inf")
            yi = points[i][1]
            for j in range(i + 1, n):
                yj = points[j][1]
                if yj <= yi and yj > max_y:
                    ans += 1
                    max_y = yj
        return ans


def _test():
    s = Solution()
    assert s.numberOfPairs([[1, 1], [2, 2], [3, 3]]) == 0
    assert s.numberOfPairs([[6, 2], [4, 4], [2, 6]]) == 2
    assert s.numberOfPairs([[3, 1], [1, 3], [1, 1]]) == 2
    print("LC 3027 OK")


if __name__ == "__main__":
    _test()
