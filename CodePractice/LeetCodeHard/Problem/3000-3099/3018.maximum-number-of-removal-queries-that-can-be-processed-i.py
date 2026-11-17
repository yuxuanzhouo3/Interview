"""
LC 3018. Maximum Number of Removal Queries That Can Be Processed I — Hard
https://leetcode.com/problems/maximum-number-of-removal-queries-that-can-be-processed-i/
Best: Interval DP — O(n^2)
"""

from typing import List


class Solution:
    def maximumProcessableQueries(self, nums: List[int], queries: List[int]) -> int:
        n, m = len(nums), len(queries)
        f = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n - 1, i - 1, -1):
                if i:
                    prev = f[i - 1][j]
                    f[i][j] = max(f[i][j], prev + (nums[i - 1] >= queries[prev] if prev < m else 0))
                if j + 1 < n:
                    prev = f[i][j + 1]
                    f[i][j] = max(f[i][j], prev + (nums[j + 1] >= queries[prev] if prev < m else 0))
                if f[i][j] == m:
                    return m
        ans = 0
        for i in range(n):
            prev = f[i][i]
            ans = max(ans, prev + (nums[i] >= queries[prev] if prev < m else 0))
        return ans


def _test():
    s = Solution()
    assert s.maximumProcessableQueries([1, 2, 3], [2, 2, 2]) == 2
    assert s.maximumProcessableQueries([2, 3], [2, 3, 2]) == 2
    print("LC 3018 OK")


if __name__ == "__main__":
    _test()
