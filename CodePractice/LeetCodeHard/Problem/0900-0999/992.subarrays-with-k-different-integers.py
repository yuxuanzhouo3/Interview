"""
LC 992. Subarrays with K Different Integers — Hard
https://leetcode.com/problems/subarrays-with-k-different-integers/
Best: atMost(k)-atMost(k-1) sliding window — O(n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def at_most(K):
            cnt = defaultdict(int)
            left = 0
            ans = 0
            for right, x in enumerate(nums):
                cnt[x] += 1
                while len(cnt) > K:
                    y = nums[left]
                    cnt[y] -= 1
                    if cnt[y] == 0:
                        del cnt[y]
                    left += 1
                ans += right - left + 1
            return ans

        return at_most(k) - at_most(k - 1)


def _test():
    s = Solution()
    assert s.subarraysWithKDistinct([1, 2, 1, 2, 3], 2) == 7
    assert s.subarraysWithKDistinct([1, 2, 1, 3, 4], 3) == 3
    print("LC 992 OK")


if __name__ == "__main__":
    _test()
