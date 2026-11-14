"""
LC 76. Minimum Window Substring — Hard
Best: Sliding window + need/have counters — O(|s|+|t|)
"""

from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = Counter(t)
        missing = len(t)
        best_l = 0
        best_len = float("inf")
        l = 0
        for r, c in enumerate(s):
            if need[c] > 0:
                missing -= 1
            need[c] -= 1
            while missing == 0:
                if r - l + 1 < best_len:
                    best_len = r - l + 1
                    best_l = l
                need[s[l]] += 1
                if need[s[l]] > 0:
                    missing += 1
                l += 1
        return "" if best_len == float("inf") else s[best_l:best_l + best_len]


def _test():
    s = Solution()
    assert s.minWindow("ADOBECODEBANC", "ABC") == "BANC"
    assert s.minWindow("a", "a") == "a"
    assert s.minWindow("a", "aa") == ""
    print("LC 76 OK")


if __name__ == "__main__":
    _test()
