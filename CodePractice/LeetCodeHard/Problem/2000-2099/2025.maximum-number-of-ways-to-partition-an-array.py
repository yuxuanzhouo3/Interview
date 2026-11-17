"""
LC 2025. Maximum Number of Ways to Partition an Array — Hard
https://leetcode.com/problems/maximum-number-of-ways-to-partition-an-array/
Best: Prefix change one index + hashmap counts — O(n)
"""


from typing import List
from collections import Counter


class Solution:
    def waysToPartition(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pref = [0] * n
        pref[0] = nums[0]
        for i in range(1, n):
            pref[i] = pref[i - 1] + nums[i]
        total = pref[-1]
        # no change
        ans = 0
        for i in range(n - 1):
            if pref[i] * 2 == total:
                ans += 1
        # change nums[j] to k, delta = k - nums[j]
        # left sum for pivot i (<n-1): if j<=i: pref[i]+delta; else pref[i]
        # need 2*left == total+delta
        right = Counter()
        for i in range(n - 1):
            right[pref[i]] += 1
        left = Counter()
        best = ans
        for j in range(n):
            delta = k - nums[j]
            # pivots i with i < j use left counts? 
            # After change, for pivot i:
            # if j <= i: need 2*(pref[i]+delta)==total+delta => 2*pref[i]+delta==total
            # if j > i: need 2*pref[i]==total+delta
            cur = 0
            if delta == 0:
                cur = ans
            else:
                # j > i => i in 0..j-1: left has pref of those pivots after processing
                # we'll maintain left = pref counts for i < j, right = pref for i >= j
                target1 = total - delta  # 2*pref == total+delta => pref == (total+delta)/2
                if (total + delta) % 2 == 0:
                    cur += left[(total + delta) // 2]
                # j <= i => pref in right (i>=j): 2*pref + delta == total => 2*pref == total-delta
                if (total - delta) % 2 == 0:
                    cur += right[(total - delta) // 2]
            best = max(best, cur)
            # move pivot j from right to left if j < n-1
            if j < n - 1:
                left[pref[j]] += 1
                right[pref[j]] -= 1
        return best


def _test():
    s = Solution()
    assert s.waysToPartition([2, -1, 2], 3) == 1
    assert s.waysToPartition([0, 0, 0], 1) == 2
    assert s.waysToPartition([22, 4, -25, -20, -15, 15, -16, 7, 19, -10, 0, -13, -14], -33) == 4
    print("LC 2025 OK")


if __name__ == "__main__":
    _test()

