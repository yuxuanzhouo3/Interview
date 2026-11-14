"""
LC 1235. Maximum Profit in Job Scheduling — Hard
https://leetcode.com/problems/maximum-profit-in-job-scheduling/
Best: Sort by end + binary search DP — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        jobs = sorted(zip(endTime, startTime, profit))
        n = len(jobs)
        ends = [e for e, _, _ in jobs]
        dp = [0] * (n + 1)
        for i, (e, s, p) in enumerate(jobs):
            j = bisect.bisect_right(ends, s, hi=i)
            dp[i + 1] = max(dp[i], dp[j] + p)
        return dp[n]


def _test():
    s = Solution()
    assert s.jobScheduling([1, 2, 3, 3], [3, 4, 5, 6], [50, 10, 40, 70]) == 120
    assert s.jobScheduling([1, 2, 3, 4, 6], [3, 5, 10, 6, 9], [20, 20, 100, 70, 60]) == 150
    assert s.jobScheduling([1, 1, 1], [2, 3, 4], [5, 6, 4]) == 6
    print("LC 1235 OK")


if __name__ == "__main__":
    _test()
