"""
LC 2472. Maximum Number of Non-Overlapping Palindrome Substrings — Hard
https://leetcode.com/problems/maximum-number-of-non-overlapping-palindrome-substrings/
Best: Greedy take shortest palindrome ≥k from left — O(n^2)
"""


class Solution:
    def maxPalindromes(self, s: str, k: int) -> int:
        n = len(s)

        def is_pal(l: int, r: int) -> bool:
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        ans = i = 0
        while i <= n - k:
            if is_pal(i, i + k - 1):
                ans += 1
                i += k
            elif i + k < n and is_pal(i, i + k):
                ans += 1
                i += k + 1
            else:
                i += 1
        return ans


def _test():
    s = Solution()
    assert s.maxPalindromes("abaccdbbd", 3) == 2
    assert s.maxPalindromes("adbcda", 2) == 0
    assert s.maxPalindromes("aaa", 1) == 3
    print("LC 2472 OK")


if __name__ == "__main__":
    _test()
