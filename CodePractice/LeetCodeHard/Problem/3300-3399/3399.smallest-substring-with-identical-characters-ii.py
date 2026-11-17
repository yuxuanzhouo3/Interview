"""
LC 3399. Smallest Substring With Identical Characters II — Hard
https://leetcode.com/problems/smallest-substring-with-identical-characters-ii/
Best: Binary search on max run length — O(n log n)
"""


class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)

        def ok(mid: int) -> bool:
            if mid == 0:
                return False
            # need to break runs so each identical run <= mid, using <= numOps flips
            ops = 0
            i = 0
            while i < n:
                j = i
                while j < n and s[j] == s[i]:
                    j += 1
                length = j - i
                ops += length // (mid + 1)
                i = j
            return ops <= numOps

        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.minLength("000001", 1) == 2
    assert s.minLength("0000", 2) == 1
    assert s.minLength("0101", 0) == 1
    print("LC 3399 OK")


if __name__ == "__main__":
    _test()
