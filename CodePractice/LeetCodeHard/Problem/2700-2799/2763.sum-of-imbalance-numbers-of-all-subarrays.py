"""
LC 2763. Sum of Imbalance Numbers of All Subarrays — Hard
https://leetcode.com/problems/sum-of-imbalance-numbers-of-all-subarrays/
Best: Enumerate subarrays with set — O(n^2)
"""

from typing import List


class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            seen = set()
            imbalance = 0
            for j in range(i, n):
                x = nums[j]
                if x not in seen:
                    # adding x into sorted unique set
                    has_l = (x - 1) in seen
                    has_r = (x + 1) in seen
                    if not seen:
                        pass
                    elif has_l and has_r:
                        imbalance -= 1
                    elif not has_l and not has_r:
                        imbalance += 1
                    # else adjacent one side: imbalance unchanged
                    seen.add(x)
                ans += imbalance
        return ans


def _test():
    s = Solution()
    assert s.sumImbalanceNumbers([2, 3, 1, 4]) == 3
    assert s.sumImbalanceNumbers([1, 3, 3, 3, 5]) == 8
    print("LC 2763 OK")


if __name__ == "__main__":
    _test()

