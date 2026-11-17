"""
LC 3017. Count the Number of Houses at a Certain Distance II — Hard
https://leetcode.com/problems/count-the-number-of-houses-at-a-certain-distance-ii/
Best: Pairwise min of line/bridge distances — O(n^2) clear; O(n) via diff possible
"""

from typing import List


class Solution:
    def countOfPairs(self, n: int, x: int, y: int) -> List[int]:
        if x > y:
            x, y = y, x
        ans = [0] * (n - 1)
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                d = min(
                    j - i,
                    abs(i - x) + abs(j - y) + 1,
                    abs(i - y) + abs(j - x) + 1,
                )
                ans[d - 1] += 2
        return ans


def _test():
    s = Solution()
    assert s.countOfPairs(3, 1, 3) == [6, 0]
    assert s.countOfPairs(5, 2, 4) == [10, 8, 2, 0]
    assert s.countOfPairs(4, 1, 1) == [6, 4, 2]
    print("LC 3017 OK")


if __name__ == "__main__":
    _test()
