"""
LC 727. Minimum Window Subsequence — Hard
https://leetcode.com/problems/minimum-window-subsequence/
Best: Forward-backward two pointers — O(n·m)
"""


class Solution:
    def minWindow(self, s1: str, s2: str) -> str:
        n, m = len(s1), len(s2)
        best_l = 0
        best_len = float("inf")
        i = 0
        while i < n:
            j = 0
            while i < n and j < m:
                if s1[i] == s2[j]:
                    j += 1
                i += 1
            if j < m:
                break
            end = i - 1
            j = m - 1
            k = end
            while j >= 0:
                if s1[k] == s2[j]:
                    j -= 1
                k -= 1
            start = k + 1
            if end - start + 1 < best_len:
                best_len = end - start + 1
                best_l = start
            i = start + 1
        return "" if best_len == float("inf") else s1[best_l : best_l + best_len]


def _test():
    s = Solution()
    assert s.minWindow("abcdebdde", "bde") == "bcde"
    assert s.minWindow("jmeqksfrsdcabefqwgeblubgxbxityggonvzctmxkdxfwweeyuqvdxtdgextkrxwappsvyfoupctohbwvuxpecsf", "gsaybiepa") == ""
    print("LC 727 OK")


if __name__ == "__main__":
    _test()
