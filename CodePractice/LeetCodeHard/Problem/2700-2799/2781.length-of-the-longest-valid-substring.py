"""
LC 2781. Length of the Longest Valid Substring — Hard
https://leetcode.com/problems/length-of-the-longest-valid-substring/
Best: Right-to-left sliding forbidden check (len≤10) — O(n * L^2)
"""

from typing import List


class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        ban = set(forbidden)
        n = len(word)
        ans = 0
        right = n
        for i in range(n - 1, -1, -1):
            for j in range(i, min(i + 10, right)):
                if word[i : j + 1] in ban:
                    right = j
                    break
            ans = max(ans, right - i)
        return ans


def _test():
    s = Solution()
    assert s.longestValidSubstring("cbaaaabc", ["aaa", "cb"]) == 4
    assert s.longestValidSubstring("leetcode", ["de", "le", "e"]) == 4
    print("LC 2781 OK")


if __name__ == "__main__":
    _test()
