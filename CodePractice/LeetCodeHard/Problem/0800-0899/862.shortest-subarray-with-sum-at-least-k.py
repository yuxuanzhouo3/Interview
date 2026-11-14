"""
LC 862. Shortest Subarray with Sum at Least K — Hard
Best: Prefix + mono deque — O(n)
"""

from typing import List
from collections import deque


class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pref = [0] * (n + 1)
        for i, x in enumerate(nums):
            pref[i + 1] = pref[i] + x
        dq = deque()
        ans = n + 1
        for i, s in enumerate(pref):
            while dq and s - pref[dq[0]] >= k:
                ans = min(ans, i - dq.popleft())
            while dq and pref[dq[-1]] >= s:
                dq.pop()
            dq.append(i)
        return ans if ans <= n else -1


def _test():
    s = Solution()
    assert s.shortestSubarray([1], 1) == 1
    assert s.shortestSubarray([1, 2], 4) == -1
    assert s.shortestSubarray([2, -1, 2], 3) == 3
    print("LC 862 OK")


if __name__ == "__main__":
    _test()
