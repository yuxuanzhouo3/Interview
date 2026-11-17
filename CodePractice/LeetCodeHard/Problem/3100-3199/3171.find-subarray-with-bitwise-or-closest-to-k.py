"""
LC 3171. Find Subarray With Bitwise OR Closest to K — Hard
https://leetcode.com/problems/find-subarray-with-bitwise-or-closest-to-k/
Best: Maintain OR-suffix sets (at most 32) — O(n * 32)
"""

from typing import List


class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        ans = abs(nums[0] - k)
        prev = {nums[0]}
        for x in nums[1:]:
            cur = {x}
            for p in prev:
                cur.add(p | x)
            for v in cur:
                ans = min(ans, abs(v - k))
            prev = cur
            if ans == 0:
                return 0
        return ans


def _test():
    s = Solution()
    assert s.minimumDifference([1, 2, 4, 5], 3) == 0
    assert s.minimumDifference([1, 3, 1, 3], 2) == 1
    assert s.minimumDifference([1], 10) == 9
    print("LC 3171 OK")


if __name__ == "__main__":
    _test()
