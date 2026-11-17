"""
LC 906. Super Palindromes — Hard
https://leetcode.com/problems/super-palindromes/
Best: Enumerate palindrome roots up to 1e9 — O(W^{1/4})
"""


class Solution:
    def superpalindromesInRange(self, left: str, right: str) -> int:
        L, R = int(left), int(right)
        # sqrt range ~ [1, 1e9]; palindrome roots
        ans = 0

        def is_pal(x: int) -> bool:
            s = str(x)
            return s == s[::-1]

        # odd-length palindromes from prefix 1..1e5
        for k in range(1, 100000):
            s = str(k)
            t = s + s[-2::-1]
            v = int(t)
            sq = v * v
            if sq > R:
                break
            if sq >= L and is_pal(sq):
                ans += 1
        # even-length
        for k in range(1, 100000):
            s = str(k)
            t = s + s[::-1]
            v = int(t)
            sq = v * v
            if sq > R:
                break
            if sq >= L and is_pal(sq):
                ans += 1
        return ans


def _test():
    s = Solution()
    assert s.superpalindromesInRange("4", "1000") == 4
    assert s.superpalindromesInRange("1", "2") == 1
    print("LC 906 OK")


if __name__ == "__main__":
    _test()
