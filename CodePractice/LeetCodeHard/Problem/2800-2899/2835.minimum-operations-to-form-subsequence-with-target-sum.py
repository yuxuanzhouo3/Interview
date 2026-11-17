"""
LC 2835. Minimum Operations to Form Subsequence With Target Sum — Hard
https://leetcode.com/problems/minimum-operations-to-form-subsequence-with-target-sum/
Best: Count powers of two + greedy split — O(n + log target · 32)
"""

from typing import List


class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        cnt = [0] * 32
        for x in nums:
            cnt[x.bit_length() - 1] += 1
        ans = 0
        i = 0
        while i < 31:
            if target & (1 << i):
                if cnt[i]:
                    cnt[i] -= 1
                else:
                    j = i + 1
                    while j < 31 and cnt[j] == 0:
                        j += 1
                    if j == 31:
                        return -1
                    while j > i:
                        cnt[j] -= 1
                        cnt[j - 1] += 2
                        ans += 1
                        j -= 1
                    cnt[i] -= 1
            cnt[i + 1] += cnt[i] // 2
            i += 1
        return ans


def _test():
    s = Solution()
    assert s.minOperations([1, 2, 8], 7) == 1
    assert s.minOperations([1, 32, 1, 2], 12) == 2
    assert s.minOperations([1, 32, 1], 35) == -1
    print("LC 2835 OK")


if __name__ == "__main__":
    _test()
