"""
LC 3911. K-th Smallest Remaining Even Integer in Subarray Queries — Hard
https://leetcode.com/problems/k-th-smallest-remaining-even-integer-in-subarray-queries/
Best: for each query binary search answer with count of evens in range removed
"""
import bisect


class Solution:
    def kthRemainingInteger(self, nums: list[int], queries: list[list[int]]) -> list[int]:
        # nums strictly increasing. Evens in subarray can be listed.
        # Positive evens: 2,4,6,... After removing those present in nums[l..r], find k-th.
        # Binary search on m = answer/2: count of positive integers ≤ m that are NOT equal to some even/2 in range.
        # I.e. among 1..m, subtract how many values v with 2v in nums[l..r].
        evens = []
        even_pos = []  # index in nums
        for i, x in enumerate(nums):
            if x % 2 == 0:
                evens.append(x // 2)  # store half
                even_pos.append(i)
        ans = []
        for l, r, k in queries:
            # evens with indices in [l,r]
            left = bisect.bisect_left(even_pos, l)
            right = bisect.bisect_right(even_pos, r)
            removed = evens[left:right]  # sorted halves present

            lo, hi = 1, 10**18
            res = hi
            while lo <= hi:
                mid = (lo + hi) // 2
                # count of remaining values among 1..mid (corresponding to even 2..2mid)
                # = mid - (# of removed halves ≤ mid)
                cnt_rem = bisect.bisect_right(removed, mid)
                remain = mid - cnt_rem
                if remain >= k:
                    res = mid * 2
                    hi = mid - 1
                else:
                    lo = mid + 1
            ans.append(res)
        return ans


def _test():
    s = Solution()
    assert s.kthRemainingInteger([1, 4, 7], [[0, 2, 1], [1, 1, 2], [0, 0, 3]]) == [2, 6, 6]
    assert s.kthRemainingInteger([2, 5, 8], [[0, 1, 2], [1, 2, 1], [0, 2, 4]]) == [6, 2, 12]
    assert s.kthRemainingInteger([3, 6], [[0, 1, 1], [1, 1, 3]]) == [2, 8]
    print("LC 3911 OK")


if __name__ == "__main__":
    _test()
