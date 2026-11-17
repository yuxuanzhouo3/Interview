"""
LC 3329. Count Substrings With K-Frequency Characters II — Hard
https://leetcode.com/problems/count-substrings-with-k-frequency-characters-ii/
Best: Sliding window — O(n)
"""

from collections import defaultdict


class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        cnt = defaultdict(int)
        left = 0
        ans = 0
        # Maintain the longest suffix ending at right that is NOT good.
        # left = first index such that s[left..right] is good (has some char freq >= k).
        # If no such, left stays > right conceptually.
        for right, ch in enumerate(s):
            cnt[ch] += 1
            while max(cnt.values()) >= k:
                cnt[s[left]] -= 1
                left += 1
            # Now s[left..right] is NOT good (or empty). All starts in [0, left-1] with end=right ARE good.
            ans += left
        return ans


def _test():
    s = Solution()
    assert s.numberOfSubstrings("abc", 2) == 0
    assert s.numberOfSubstrings("aa", 2) == 1
    assert s.numberOfSubstrings("aaa", 2) == 3
    assert s.numberOfSubstrings("abacb", 2) == 4
    print("LC 3329 OK")


if __name__ == "__main__":
    _test()
