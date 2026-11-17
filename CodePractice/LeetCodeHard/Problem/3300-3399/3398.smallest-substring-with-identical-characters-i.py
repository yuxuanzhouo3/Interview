"""
LC 3398. Smallest Substring With Identical Characters I — Hard
https://leetcode.com/problems/smallest-substring-with-identical-characters-i/
Best: 方法一
"""

class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        def check(m: int) -> bool:
            cnt = 0
            if m == 1:
                t = "01"
                cnt = sum(c == t[i & 1] for i, c in enumerate(s))
                cnt = min(cnt, n - cnt)
            else:
                k = 0
                for i, c in enumerate(s):
                    k += 1
                    if i == len(s) - 1 or c != s[i + 1]:
                        cnt += k // (m + 1)
                        k = 0
            return cnt <= numOps

        n = len(s)
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if check(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

def _test():
    s = Solution()
    assert s.minLength(s='000001', numOps=1) == 2
    assert s.minLength(s='0000', numOps=2) == 1
    assert s.minLength(s='0101', numOps=0) == 1
    print("LC 3398 OK")

if __name__ == "__main__":
    _test()
