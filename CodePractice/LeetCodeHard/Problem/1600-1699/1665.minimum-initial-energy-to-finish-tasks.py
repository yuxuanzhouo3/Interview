"""
LC 1665. Minimum Initial Energy to Finish Tasks — Hard
https://leetcode.com/problems/minimum-initial-energy-to-finish-tasks/
Best: sort by (minimum-actual) desc; greedy accumulate — O(n log n)
"""

from typing import List


class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        # actual, minimum — sort by (minimum - actual) descending
        tasks.sort(key=lambda t: t[1] - t[0], reverse=True)
        energy = 0
        cur = 0
        for actual, minimum in tasks:
            if cur < minimum:
                energy += minimum - cur
                cur = minimum
            cur -= actual
        return energy


def _test():
    s = Solution()
    assert s.minimumEffort([[1, 2], [2, 4], [4, 8]]) == 8
    assert s.minimumEffort([[1, 3], [2, 4], [10, 11], [10, 12], [8, 9]]) == 32
    assert s.minimumEffort([[1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]]) == 27
    print("LC 1665 OK")


if __name__ == "__main__":
    _test()
