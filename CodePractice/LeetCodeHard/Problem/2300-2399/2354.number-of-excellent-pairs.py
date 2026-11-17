"""
LC 2354. Number of Excellent Pairs — Hard
https://leetcode.com/problems/number-of-excellent-pairs/
Best: bits(a|b)+bits(a&b)=bits(a)+bits(b); count by popcount — O(n·W)
"""

from typing import List


class Solution:
    def countExcellentPairs(self, nums: List[int], k: int) -> int:
        uniq = set(nums)
        cnt = [0] * 31
        for x in uniq:
            cnt[bin(x).count("1")] += 1
        ans = 0
        for i in range(31):
            for j in range(31):
                if i + j >= k:
                    ans += cnt[i] * cnt[j]
        return ans


def _test():
    s = Solution()
    assert s.countExcellentPairs([1, 2, 3, 1], 3) == 5
    assert s.countExcellentPairs([5, 1, 1], 10) == 0
    print("LC 2354 OK")


if __name__ == "__main__":
    _test()
