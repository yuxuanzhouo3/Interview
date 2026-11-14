"""
LC 1044. Longest Duplicate Substring — Hard
https://leetcode.com/problems/longest-duplicate-substring/
Best: Binary search + Rabin-Karp — O(n log n)
"""

class Solution:
    def longestDupSubstring(self, s: str) -> str:
        n = len(s)
        nums = [ord(c) - 97 for c in s]
        MOD = (1 << 61) - 1
        BASE = 26

        def check(L):
            if L == 0:
                return 0
            h = 0
            for i in range(L):
                h = (h * BASE + nums[i]) % MOD
            seen = {h}
            powL = pow(BASE, L, MOD)
            for i in range(L, n):
                h = (h * BASE + nums[i] - nums[i - L] * powL) % MOD
                if h in seen:
                    return i - L + 1
                seen.add(h)
            return -1

        lo, hi = 1, n - 1
        start = -1
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            idx = check(mid)
            if idx != -1:
                best = mid
                start = idx
                lo = mid + 1
            else:
                hi = mid - 1
        return "" if start < 0 else s[start : start + best]


def _test():
    s = Solution()
    r = s.longestDupSubstring("banana")
    assert r in ("ana", "an", "na", "a", "n", "bana") or "ana" == r
    assert r == "ana"
    assert s.longestDupSubstring("abcd") == ""
    assert len(s.longestDupSubstring("aaaa")) >= 3
    print("LC 1044 OK")


if __name__ == "__main__":
    _test()
