"""
LC 1183. Maximum Number of Ones — Hard
https://leetcode.com/problems/maximum-number-of-ones/
Tags: Math, Greedy
Best: 按 sideLength 瓦片位置出现频次取前 maxOnes
"""

from typing import List
import heapq


class Solution:
    def maximumNumberOfOnes(self, width: int, height: int, sideLength: int, maxOnes: int) -> int:
        counts = []
        for i in range(sideLength):
            for j in range(sideLength):
                rows = (height - i + sideLength - 1) // sideLength
                cols = (width - j + sideLength - 1) // sideLength
                counts.append(rows * cols)
        counts.sort(reverse=True)
        return sum(counts[:maxOnes])


def _test():
    sol = Solution()
    assert sol.maximumNumberOfOnes(3, 3, 2, 1) == 4
    assert sol.maximumNumberOfOnes(3, 3, 2, 2) == 6
    print("LC 1183 OK")


if __name__ == "__main__":
    _test()

