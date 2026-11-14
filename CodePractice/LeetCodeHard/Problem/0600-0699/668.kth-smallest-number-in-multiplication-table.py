"""
LC 668. Kth Smallest Number in Multiplication Table — Hard
https://leetcode.com/problems/kth-smallest-number-in-multiplication-table/
Best: binary search on value — O(m log(m*n))
"""


class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        def count_le(x: int) -> int:
            cnt = 0
            for i in range(1, m + 1):
                cnt += min(n, x // i)
            return cnt

        lo, hi = 1, m * n
        while lo < hi:
            mid = (lo + hi) // 2
            if count_le(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.findKthNumber(3, 3, 5) == 3
    assert s.findKthNumber(2, 3, 6) == 6
    print("LC 668 OK")


if __name__ == "__main__":
    _test()
