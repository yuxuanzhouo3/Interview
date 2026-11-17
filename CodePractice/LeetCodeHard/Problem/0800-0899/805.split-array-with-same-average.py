"""
LC 805. Split Array With Same Average — Hard
Best: Meet-in-middle / DP subset sums by size — O(n*sum)
"""

from typing import List


class Solution:
    def splitArraySameAverage(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return False
        total = sum(nums)
        nums = [x * n - total for x in nums]  # transform: need subset sum 0, size 1..n-1
        # equivalent: subset with sum*n == total*size
        A = nums[: n // 2]
        B = nums[n // 2 :]

        def sums(arr):
            res = {0: {0}}
            for x in arr:
                nxt = {}
                for s, sizes in res.items():
                    nxt.setdefault(s, set()).update(sizes)
                    for sz in sizes:
                        nxt.setdefault(s + x, set()).add(sz + 1)
                res = nxt
            return res

        left = sums(A)
        right = sums(B)
        # need sumL + sumR = 0 and size not 0 or n
        for s, sizes in left.items():
            if -s in right:
                for a in sizes:
                    for b in right[-s]:
                        if 0 < a + b < n:
                            return True
        return False


def _test():
    s = Solution()
    assert s.splitArraySameAverage([1, 2, 3, 4, 5, 6, 7, 8]) is True
    assert s.splitArraySameAverage([3, 1]) is False
    print("LC 805 OK")


if __name__ == "__main__":
    _test()
