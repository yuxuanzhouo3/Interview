"""
LC 995. Minimum Number of K Consecutive Bit Flips — Hard
https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/
Best: Diff / queue flip window — O(n)
"""

from typing import List
from collections import deque


class Solution:
    def minKBitFlips(self, nums: List[int], k: int) -> int:
        n = len(nums)
        q = deque()
        ans = 0
        for i in range(n):
            while q and q[0] + k <= i:
                q.popleft()
            # current effective bit
            bit = nums[i] ^ (len(q) % 2)
            if bit == 0:
                if i + k > n:
                    return -1
                q.append(i)
                ans += 1
        return ans


def _test():
    s = Solution()
    assert s.minKBitFlips([0, 1, 0], 1) == 2
    assert s.minKBitFlips([1, 1, 0], 2) == -1
    assert s.minKBitFlips([0, 0, 0, 1, 0, 1, 1, 0], 3) == 3
    print("LC 995 OK")


if __name__ == "__main__":
    _test()
