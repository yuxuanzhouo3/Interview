"""
LC 2071. Maximum Number of Tasks You Can Assign — Hard
https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/
Best: binary search + greedy deque — O(n log n log m)
"""

from typing import List
from collections import deque
import bisect


class Solution:
    def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        tasks.sort()
        workers.sort()

        def can(k):
            # k hardest tasks among smallest? take k easiest tasks vs k strongest workers
            t = tasks[:k]
            w = workers[-k:]
            p = pills
            dq = deque()
            j = 0
            for worker in w:
                while j < k and t[j] <= worker + strength:
                    dq.append(t[j])
                    j += 1
                if not dq:
                    return False
                if dq[0] <= worker:
                    dq.popleft()
                else:
                    if p == 0:
                        return False
                    p -= 1
                    dq.pop()  # use pill on hardest feasible
            return True

        lo, hi = 0, min(len(tasks), len(workers))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


def _test():
    s = Solution()
    assert s.maxTaskAssign([3, 2, 1], [0, 3, 3], 1, 1) == 3
    assert s.maxTaskAssign([5, 4], [0, 0, 0], 1, 5) == 1
    assert s.maxTaskAssign([10, 15, 30], [0, 10, 10, 10, 10], 3, 10) == 2
    print("LC 2071 OK")


if __name__ == "__main__":
    _test()
