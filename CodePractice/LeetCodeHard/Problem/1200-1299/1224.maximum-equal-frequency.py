"""
LC 1224. Maximum Equal Frequency — Hard
https://leetcode.com/problems/maximum-equal-frequency/
Best: Frequency of frequencies — O(n)
"""

from collections import defaultdict
from typing import List


class Solution:
    def maxEqualFreq(self, nums: List[int]) -> int:
        cnt = defaultdict(int)
        freq = defaultdict(int)
        ans = 0
        maxf = 0
        for i, x in enumerate(nums, 1):
            if cnt[x]:
                freq[cnt[x]] -= 1
            cnt[x] += 1
            freq[cnt[x]] += 1
            maxf = max(maxf, cnt[x])
            # cases: all freq 1; one number appears maxf and others maxf-1 with (maxf-1)*count+1==i
            # or one number appears once and others share same freq
            if (
                maxf == 1
                or (freq[maxf] * maxf + freq[1] == i and freq[1] == 1)
                or (freq[maxf] == 1 and freq[maxf - 1] * (maxf - 1) + maxf == i)
            ):
                ans = i
        return ans


def _test():
    s = Solution()
    assert s.maxEqualFreq([2, 2, 1, 1, 5, 3, 3, 5]) == 7
    assert s.maxEqualFreq([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]) == 13
    assert s.maxEqualFreq([1, 1]) == 2
    assert s.maxEqualFreq([10, 2, 8, 9, 3, 8, 1, 5, 2, 3, 7, 6]) == 8
    print("LC 1224 OK")


if __name__ == "__main__":
    _test()
