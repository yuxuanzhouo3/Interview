"""
LC 1520. Maximum Number of Non-Overlapping Substrings — Hard
https://leetcode.com/problems/maximum-number-of-non-overlapping-substrings/
Best: Expand char ranges then greedy by end — O(n)
"""

from typing import List


class Solution:
    def maxNumOfSubstrings(self, s: str) -> List[str]:
        first = [-1] * 26
        last = [-1] * 26
        for i, ch in enumerate(s):
            c = ord(ch) - 97
            if first[c] < 0:
                first[c] = i
            last[c] = i

        intervals = []
        for c in range(26):
            if first[c] < 0:
                continue
            l, r = first[c], last[c]
            i = l
            ok = True
            while i <= r:
                cc = ord(s[i]) - 97
                if first[cc] < l:
                    ok = False
                    break
                r = max(r, last[cc])
                i += 1
            if ok:
                intervals.append((r, l))

        intervals.sort()
        ans = []
        prev_end = -1
        for r, l in intervals:
            if l > prev_end:
                ans.append(s[l : r + 1])
                prev_end = r
        return ans


def _test():
    s = Solution()

    def valid_answer(s0, parts):
        # non-overlapping, each part is a valid closed char-range substring
        first, last = {}, {}
        for i, ch in enumerate(s0):
            first.setdefault(ch, i)
            last[ch] = i
        used = []
        for p in parts:
            i = s0.find(p)
            assert i >= 0
            j = i + len(p) - 1
            for a, b in used:
                assert b < i or j < a
            used.append((i, j))
            for ch in set(p):
                assert first[ch] >= i and last[ch] <= j

    r1 = s.maxNumOfSubstrings("adefaddacacac")
    assert len(r1) == 2  # max is ["e","f"] (official sample text is outdated)
    valid_answer("adefaddacacac", r1)
    r2 = s.maxNumOfSubstrings("abbaccd")
    assert len(r2) == 3
    valid_answer("abbaccd", r2)
    print("LC 1520 OK")


if __name__ == "__main__":
    _test()
