"""
LC 363. Max Sum of Rectangle No Larger Than K — Hard
Best: Enumerate left-right cols + sorted prefix — O(min(m,n)^2 * max * log)
"""

from typing import List
import bisect


class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])
        ans = float("-inf")
        for left in range(n):
            row_sum = [0] * m
            for right in range(left, n):
                for i in range(m):
                    row_sum[i] += matrix[i][right]
                # max subarray sum <= k via sorted prefix
                prefix = [0]
                s = 0
                for x in row_sum:
                    s += x
                    # find smallest prefix >= s-k
                    j = bisect.bisect_left(prefix, s - k)
                    if j < len(prefix):
                        ans = max(ans, s - prefix[j])
                    bisect.insort(prefix, s)
        return ans


def _test():
    s = Solution()
    assert s.maxSumSubmatrix([[1, 0, 1], [0, -2, 3]], 2) == 2
    assert s.maxSumSubmatrix([[2, 2, -1]], 3) == 3
    print("LC 363 OK")


if __name__ == "__main__":
    _test()
