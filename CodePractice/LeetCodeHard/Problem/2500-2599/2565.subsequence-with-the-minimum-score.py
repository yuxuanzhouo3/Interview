"""
LC 2565. Subsequence With the Minimum Score — Hard
https://leetcode.com/problems/subsequence-with-the-minimum-score/
Best: Prefix/suffix match two pointers — O(n)
"""

class Solution:
    def minimumScore(self, s: str, t: str) -> int:
        n, m = len(s), len(t)
        # leftmost position in s to match t[0..i]
        left = [-1] * m
        j = 0
        for i in range(n):
            if j < m and s[i] == t[j]:
                left[j] = i
                j += 1
        if j == m:
            return 0
        # rightmost position in s to match t[i..]
        right = [-1] * m
        j = m - 1
        for i in range(n - 1, -1, -1):
            if j >= 0 and s[i] == t[j]:
                right[j] = i
                j -= 1
        # remove t[l..r] (inclusive), need left match t[0..l) and right match t[r+1..]
        ans = m
        # remove prefix
        for r in range(m):
            if right[r] != -1:
                ans = min(ans, r)  # remove t[0..r-1]? wait remove [0..r) means keep right from r
        # actually score = length of removed contiguous subseq of t
        # can remove empty? if t is subsequence return 0 already
        # remove prefix t[0..i): need right[i] matched
        for i in range(m):
            if right[i] != -1:
                ans = min(ans, i)
        # remove suffix t[i+1..]: need left[i]
        for i in range(m):
            if left[i] != -1:
                ans = min(ans, m - 1 - i)
        # remove middle t[i+1..j-1]: left[i] < right[j]
        j = 0
        for i in range(m):
            if left[i] == -1:
                break
            while j < m and (right[j] == -1 or right[j] <= left[i]):
                j += 1
            if j < m:
                ans = min(ans, j - i - 1)
            else:
                break
        # remove all
        ans = min(ans, m)
        return ans


def _test():
    s = Solution()
    assert s.minimumScore("abacaba", "bzaa") == 1
    assert s.minimumScore("cde", "xyz") == 3
    assert s.minimumScore("aaaa", "aa") == 0
    print("LC 2565 OK")


if __name__ == "__main__":
    _test()

