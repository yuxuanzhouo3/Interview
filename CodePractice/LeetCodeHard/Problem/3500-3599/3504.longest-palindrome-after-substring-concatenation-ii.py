"""
LC 3504. Longest Palindrome After Substring Concatenation II — Hard
https://leetcode.com/problems/longest-palindrome-after-substring-concatenation-ii/
Best: 方法一：枚举回文中点 + 动态规划
"""

class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        def expand(s: str, g: list[int], l: int, r: int):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                g[l] = max(g[l], r - l + 1)
                l, r = l - 1, r + 1

        def calc(s: str) -> list[int]:
            n = len(s)
            g = [0] * n
            for i in range(n):
                expand(s, g, i, i)
                expand(s, g, i, i + 1)
            return g

        m, n = len(s), len(t)
        t = t[::-1]
        g1, g2 = calc(s), calc(t)
        ans = max(*g1, *g2)
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i, a in enumerate(s, 1):
            for j, b in enumerate(t, 1):
                if a == b:
                    f[i][j] = f[i - 1][j - 1] + 1
                    ans = max(ans, f[i][j] * 2 + (0 if i >= m else g1[i]))
                    ans = max(ans, f[i][j] * 2 + (0 if j >= n else g2[j]))
        return ans

def _test():
    s = Solution()
    assert s.longestPalindrome(s='a', t='a') == 2
    assert s.longestPalindrome(s='abc', t='def') == 1
    assert s.longestPalindrome(s='b', t='aaaa') == 4
    print("LC 3504 OK")

if __name__ == "__main__":
    _test()
