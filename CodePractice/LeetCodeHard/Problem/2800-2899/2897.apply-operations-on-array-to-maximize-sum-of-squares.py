"""
LC 2897. Apply Operations on Array to Maximize Sum of Squares — Hard
https://leetcode.com/problems/apply-operations-on-array-to-maximize-sum-of-squares/
Best: Bit greedy redistribute — O(n*30)
"""

from typing import List


class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        bits = [0] * 31
        for x in nums:
            for b in range(31):
                if x >> b & 1:
                    bits[b] += 1
        ans = 0
        for _ in range(k):
            cur = 0
            for b in range(31):
                if bits[b]:
                    bits[b] -= 1
                    cur |= 1 << b
            ans = (ans + cur * cur) % MOD
        return ans


def _test():
    s = Solution()
    assert s.maxSum([2, 6, 5, 8], 2) == 261
    assert s.maxSum([4, 5, 4, 7], 3) == 90
    print("LC 2897 OK")


if __name__ == "__main__":
    _test()

