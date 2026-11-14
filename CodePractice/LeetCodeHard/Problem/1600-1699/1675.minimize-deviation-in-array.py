"""
LC 1675. Minimize Deviation in Array — Hard
https://leetcode.com/problems/minimize-deviation-in-array/
Best: max-heap + unify odds — O(n log n log M)
"""

from typing import List
import heapq


class Solution:
    def minimumDeviation(self, nums: List[int]) -> int:
        pq = []
        mn = float("inf")
        for x in nums:
            if x % 2:
                x *= 2
            mn = min(mn, x)
            heapq.heappush(pq, -x)
        ans = float("inf")
        while True:
            mx = -heapq.heappop(pq)
            ans = min(ans, mx - mn)
            if mx % 2:
                break
            mx //= 2
            mn = min(mn, mx)
            heapq.heappush(pq, -mx)
        return ans


def _test():
    s = Solution()
    assert s.minimumDeviation([1, 2, 3, 4]) == 1
    assert s.minimumDeviation([4, 1, 5, 20, 3]) == 3
    assert s.minimumDeviation([2, 10, 8]) == 3
    print("LC 1675 OK")


if __name__ == "__main__":
    _test()
