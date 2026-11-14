"""
LC 1723. Find Minimum Time to Finish All Jobs — Hard
https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/
Best: binary search + pruned DFS — O(k^n) pruned / practical
"""

from typing import List


class Solution:
    def minimumTimeRequired(self, jobs: List[int], k: int) -> int:
        jobs.sort(reverse=True)
        lo, hi = max(jobs), sum(jobs)

        def can(limit):
            workloads = [0] * k

            def dfs(i):
                if i == len(jobs):
                    return True
                seen = set()
                for j in range(k):
                    if workloads[j] in seen:
                        continue
                    if workloads[j] + jobs[i] > limit:
                        continue
                    seen.add(workloads[j])
                    workloads[j] += jobs[i]
                    if dfs(i + 1):
                        return True
                    workloads[j] -= jobs[i]
                    if workloads[j] == 0:
                        break
                return False

            return dfs(0)

        while lo < hi:
            mid = (lo + hi) // 2
            if can(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.minimumTimeRequired([3, 2, 3], 3) == 3
    assert s.minimumTimeRequired([1, 2, 4, 7, 8], 2) == 11
    print("LC 1723 OK")


if __name__ == "__main__":
    _test()
