"""
LC 3008. Find Beautiful Indices in the Given Array II — Hard
https://leetcode.com/problems/find-beautiful-indices-in-the-given-array-ii/
Best: KMP + binary search — O(n)
"""

from typing import List
import bisect


class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        def find_all(text: str, pat: str) -> List[int]:
            m = len(pat)
            if m == 0:
                return list(range(len(text) + 1))
            lps = [0] * m
            length = 0
            i = 1
            while i < m:
                if pat[i] == pat[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                elif length:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
            res = []
            i = j = 0
            n = len(text)
            while i < n:
                if text[i] == pat[j]:
                    i += 1
                    j += 1
                    if j == m:
                        res.append(i - m)
                        j = lps[j - 1]
                elif j:
                    j = lps[j - 1]
                else:
                    i += 1
            return res

        A = find_all(s, a)
        B = find_all(s, b)
        ans = []
        for i in A:
            lo = bisect.bisect_left(B, i - k)
            if lo < len(B) and B[lo] <= i + k:
                ans.append(i)
        return ans


def _test():
    s = Solution()
    assert s.beautifulIndices("isawsquirrelnearmysquirrelhouseohmy", "my", "squirrel", 15) == [16, 33]
    assert s.beautifulIndices("abcd", "a", "a", 4) == [0]
    print("LC 3008 OK")


if __name__ == "__main__":
    _test()
