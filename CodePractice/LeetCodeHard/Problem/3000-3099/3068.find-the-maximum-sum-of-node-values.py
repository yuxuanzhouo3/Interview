"""
LC 3068. Find the Maximum Sum of Node Values — Hard
https://leetcode.com/problems/find-the-maximum-sum-of-node-values/
Best: XOR-k gains; take even count of positive / best swap — O(n)
"""

from typing import List


class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        # Choosing a set of nodes to XOR with k is valid iff even size (tree edges)
        base = sum(nums)
        gains = [(x ^ k) - x for x in nums]
        gains.sort(reverse=True)
        best = base
        cur = base
        for i in range(0, len(gains) - 1, 2):
            cur += gains[i] + gains[i + 1]
            best = max(best, cur)
        return best


def _test():
    s = Solution()
    assert s.maximumValueSum([1, 2, 1], 3, [[0, 1], [0, 2]]) == 6
    assert s.maximumValueSum([2, 3], 7, [[0, 1]]) == 9
    assert s.maximumValueSum([7, 7, 7, 7, 7, 7], 3, [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]) == 42
    print("LC 3068 OK")


if __name__ == "__main__":
    _test()
