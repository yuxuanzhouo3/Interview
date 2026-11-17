"""
LC 3579. Minimum Steps to Convert String with Operations — Hard
https://leetcode.com/problems/minimum-steps-to-convert-string-with-operations/
Best: 方法一：贪心 + 动态规划
"""

from collections import Counter
from math import inf

class Solution:
    def minOperations(self, word1: str, word2: str) -> int:
        def calc(l: int, r: int, rev: bool) -> int:
            cnt = Counter()
            res = 0
            for i in range(l, r + 1):
                j = r - (i - l) if rev else i
                a, b = word1[j], word2[i]
                if a != b:
                    if cnt[(b, a)] > 0:
                        cnt[(b, a)] -= 1
                    else:
                        cnt[(a, b)] += 1
                        res += 1
            return res

        n = len(word1)
        f = [inf] * (n + 1)
        f[0] = 0
        for i in range(1, n + 1):
            for j in range(i):
                t = min(calc(j, i - 1, False), 1 + calc(j, i - 1, True))
                f[i] = min(f[i], f[j] + t)
        return f[n]

def _test():
    s = Solution()
    assert s.minOperations(word1='abcdf', word2='dacbe') == 4
    assert s.minOperations(word1='abceded', word2='baecfef') == 4
    assert s.minOperations(word1='abcdef', word2='fedabc') == 2
    print("LC 3579 OK")

if __name__ == "__main__":
    _test()
