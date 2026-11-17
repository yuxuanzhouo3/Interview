"""
LC 3049. Earliest Second to Mark Indices II — Hard
https://leetcode.com/problems/earliest-second-to-mark-indices-ii/
Best: Binary search + first-occurrence reset greedy heap — O(m log m)
"""

from typing import List
import heapq


class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        # first second where we can reset index i (nums[i]>0)
        first = [-1] * n
        for s in range(m - 1, -1, -1):
            idx = changeIndices[s] - 1
            if nums[idx] > 0:
                first[idx] = s
        total = sum(nums) + n

        def check(t: int) -> bool:
            h = []
            cnt = 0  # free seconds for decrement / mark
            for s in range(t - 1, -1, -1):
                idx = changeIndices[s] - 1
                if s != first[idx]:
                    cnt += 1
                    continue
                # candidate reset at first occurrence
                heapq.heappush(h, nums[idx])
                if cnt > 0:
                    cnt -= 1  # reserve one later second as mark
                else:
                    # cannot mark after reset; use this second as free op instead
                    cnt += 1
                    heapq.heappop(h)
            # remaining: manual decrement+mark for indices not reset
            return total - (sum(h) + len(h)) <= cnt

        lo, hi = 1, m
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if check(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans


def _test():
    s = Solution()
    assert s.earliestSecondToMarkIndices([3, 2, 3], [1, 3, 2, 2, 2, 2, 3]) == 6
    assert s.earliestSecondToMarkIndices([0, 0, 1, 2], [1, 2, 1, 2, 1, 2, 1, 2]) == 7
    assert s.earliestSecondToMarkIndices([1, 2, 3], [1, 2, 3]) == -1
    print("LC 3049 OK")


if __name__ == "__main__":
    _test()
