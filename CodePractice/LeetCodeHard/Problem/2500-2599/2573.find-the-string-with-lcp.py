"""
LC 2573. Find the String with LCP — Hard
https://leetcode.com/problems/find-the-string-with-lcp/
Best: Greedy assign letters + verify LCP — O(n²)
"""

from string import ascii_lowercase
from typing import List

class Solution:
    def findTheString(self, lcp: List[List[int]]) -> str:
        n = len(lcp)
        s = [""] * n
        i = 0
        for c in ascii_lowercase:
            while i < n and s[i]:
                i += 1
            if i == n:
                break
            for j in range(i, n):
                if lcp[i][j]:
                    s[j] = c
        if "" in s:
            return ""
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == s[j]:
                    if i == n - 1 or j == n - 1:
                        if lcp[i][j] != 1:
                            return ""
                    elif lcp[i][j] != lcp[i + 1][j + 1] + 1:
                        return ""
                elif lcp[i][j]:
                    return ""
        return "".join(s)

def _test():

    s = Solution()
    assert s.findTheString([[4, 0, 2, 0], [0, 3, 0, 1], [2, 0, 2, 0], [0, 1, 0, 1]]) == "abab"
    assert s.findTheString([[4, 3, 2, 1], [3, 3, 2, 1], [2, 2, 2, 1], [1, 1, 1, 1]]) == "aaaa"
    assert s.findTheString([[4, 3, 2, 1], [3, 3, 2, 1], [2, 2, 2, 1], [1, 1, 1, 3]]) == ""
    print("LC 2573 OK")

if __name__ == "__main__":
    _test()
