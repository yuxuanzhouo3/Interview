"""
LC 3681. Maximum XOR of Subsequences — Hard
https://leetcode.com/problems/maximum-xor-of-subsequences/
Best: XOR linear basis (Gaussian) — O(n * 32)
"""

from typing import List


class Solution:
    def maxXorSubsequences(self, nums: List[int]) -> int:
        basis = [0] * 32
        for x in nums:
            for b in range(31, -1, -1):
                if not (x >> b) & 1:
                    continue
                if basis[b] == 0:
                    basis[b] = x
                    break
                x ^= basis[b]
        ans = 0
        for b in range(31, -1, -1):
            if ans ^ basis[b] > ans:
                ans ^= basis[b]
        return ans


def _test():
    s = Solution()
    assert s.maxXorSubsequences([1, 2, 3]) == 3
    assert s.maxXorSubsequences([5, 1, 6]) == 7
    print("LC 3681 OK")


if __name__ == "__main__":
    _test()
