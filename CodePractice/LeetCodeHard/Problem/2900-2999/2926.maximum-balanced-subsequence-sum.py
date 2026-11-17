"""
LC 2926. Maximum Balanced Subsequence Sum — Hard
https://leetcode.com/problems/maximum-balanced-subsequence-sum/
Best: Fenwick/seg max dp on nums[i]-i — O(n log n)
"""

from typing import List


class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # balanced: nums[i]-i <= nums[j]-j for i<j in subsequence
        keys = sorted(set(nums[i] - i for i in range(n)))
        rank = {v: i + 1 for i, v in enumerate(keys)}
        m = len(keys)
        bit = [-10**18] * (m + 1)

        def upd(i, v):
            while i <= m:
                bit[i] = max(bit[i], v)
                i += i & -i

        def qry(i):
            r = -10**18
            while i:
                r = max(r, bit[i])
                i -= i & -i
            return r

        ans = -10**18
        for i, x in enumerate(nums):
            r = rank[x - i]
            best = qry(r)
            cur = x if best < -10**17 else x + best
            cur = max(cur, x)
            ans = max(ans, cur)
            upd(r, cur)
        return ans


def _test():
    s = Solution()
    assert s.maxBalancedSubsequenceSum([3, 3, 5, 6]) == 14
    assert s.maxBalancedSubsequenceSum([5, -1, -3, 8]) == 13
    assert s.maxBalancedSubsequenceSum([-2, -1]) == -1
    print("LC 2926 OK")


if __name__ == "__main__":
    _test()

