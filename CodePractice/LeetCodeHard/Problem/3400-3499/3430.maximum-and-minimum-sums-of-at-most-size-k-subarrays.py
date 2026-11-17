"""
LC 3430. Maximum and Minimum Sums of at Most Size K Subarrays — Hard
https://leetcode.com/problems/maximum-and-minimum-sums-of-at-most-size-k-subarrays/
Best: Monotonic stack contribution with window size ≤ k — O(n)
"""

from typing import List


class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        def contrib(want_max: bool) -> int:
            n = len(nums)
            left = [0] * n
            right = [0] * n
            stk = []
            for i in range(n):
                while stk and (
                    nums[stk[-1]] < nums[i]
                    if want_max
                    else nums[stk[-1]] > nums[i]
                ):
                    stk.pop()
                # strict: equal goes to previous occurrence's right
                while stk and nums[stk[-1]] == nums[i]:
                    # for max: treat equal as not greater → left boundary
                    break
                left[i] = stk[-1] if stk else -1
                # fix equal handling: for maximum, previous equal should be boundary (left exclusive)
                # Use non-strict pop for opposite
                stk.append(i)
            # redo with clearer standard
            return 0

        # Standard contribution for sum of subarray minimums with length constraint
        def sum_extreme(is_min: bool) -> int:
            n = len(nums)
            # prev[i]: nearest index j < i with nums[j] < nums[i] (for min) / > for max; equal: <= / >=
            prev = [-1] * n
            nxt = [n] * n
            stk = []
            for i in range(n):
                if is_min:
                    while stk and nums[stk[-1]] >= nums[i]:
                        stk.pop()
                else:
                    while stk and nums[stk[-1]] <= nums[i]:
                        stk.pop()
                prev[i] = stk[-1] if stk else -1
                stk.append(i)
            stk = []
            for i in range(n - 1, -1, -1):
                if is_min:
                    while stk and nums[stk[-1]] > nums[i]:
                        stk.pop()
                else:
                    while stk and nums[stk[-1]] < nums[i]:
                        stk.pop()
                nxt[i] = stk[-1] if stk else n
                stk.append(i)
            total = 0
            for i in range(n):
                L = i - prev[i]
                R = nxt[i] - i
                # number of subarrays containing i as extreme with length <= k
                # subarray [l,r] with prev < l <= i <= r < nxt, and r-l+1 <= k
                # count = sum_{a=1..L} min(R, k-a+1) for a where k-a+1 > 0
                max_left = min(L, k)
                # for left span a in 1..max_left, right span can be 1..min(R, k-a+1)
                # sum_{a=1}^{max_left} min(R, k-a+1)
                # split when k-a+1 >= R i.e. a <= k-R+1
                t = 0
                # a from 1 to max_left
                # if R >= k: then min=k-a+1 for all a → sum_{a=1}^{max_left}(k-a+1)
                # else: for a <= k-R+1: contrib R; for a > k-R+1: contrib k-a+1
                threshold = k - R + 1
                a1 = min(max_left, max(0, threshold))
                # a in 1..a1: +R each (if a1>=1)
                if a1 >= 1:
                    t += a1 * R
                a_start = max(a1 + 1, 1)
                if a_start <= max_left:
                    # sum (k-a+1) for a=a_start..max_left
                    # = sum_{x=k-max_left+1 .. k-a_start+1} x
                    lo = k - max_left + 1
                    hi = k - a_start + 1
                    t += (hi + lo) * (hi - lo + 1) // 2
                total += nums[i] * t
            return total

        return sum_extreme(True) + sum_extreme(False)


def _test():
    s = Solution()
    assert s.minMaxSubarraySum([1, 2, 3], 2) == 20
    assert s.minMaxSubarraySum([1, -3, 1], 2) == -6
    print("LC 3430 OK")


if __name__ == "__main__":
    _test()
