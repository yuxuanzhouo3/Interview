"""
LC 982. Triples with Bitwise AND Equal To Zero — Hard
https://leetcode.com/problems/triples-with-bitwise-and-equal-to-zero/
Best: Count AND pairs then check — O(n^2 + n*2^16)
"""

from typing import List
from collections import Counter


class Solution:
    def countTriplets(self, nums: List[int]) -> int:
        cnt = Counter()
        for a in nums:
            for b in nums:
                cnt[a & b] += 1
        ans = 0
        for c in nums:
            for ab, v in cnt.items():
                if ab & c == 0:
                    ans += v
        return ans


def _test():
    s = Solution()
    assert s.countTriplets([2, 1, 3]) == 12
    assert s.countTriplets([0, 0, 0]) == 27
    print("LC 982 OK")


if __name__ == "__main__":
    _test()
