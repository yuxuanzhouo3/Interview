"""
LC 2488. Count Subarrays With Median K — Hard
https://leetcode.com/problems/count-subarrays-with-median-k/
Best: Balance map around k — O(n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pos = nums.index(k)
        bal = 0
        freq = defaultdict(int)
        freq[0] = 1
        for i in range(pos - 1, -1, -1):
            bal += 1 if nums[i] > k else -1
            freq[bal] += 1
        ans = 0
        bal = 0
        for i in range(pos, n):
            if i > pos:
                bal += 1 if nums[i] > k else -1
            # odd: total bal 0 → left=-bal; even left-median: total bal 1 → left=1-bal
            ans += freq[-bal] + freq[1 - bal]
        return ans


def _test():
    s = Solution()
    assert s.countSubarrays([3, 2, 1, 4, 5], 4) == 3
    assert s.countSubarrays([2, 3, 1], 3) == 1
    assert s.countSubarrays([2, 5, 1, 4, 3, 6], 1) == 3
    print("LC 2488 OK")


if __name__ == "__main__":
    _test()
