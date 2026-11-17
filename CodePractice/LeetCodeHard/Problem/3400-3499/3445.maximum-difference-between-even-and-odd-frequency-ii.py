"""
LC 3445. Maximum Difference Between Even and Odd Frequency II — Hard
https://leetcode.com/problems/maximum-difference-between-even-and-odd-frequency-ii/
Best: 方法一：枚举字符对 + 滑动窗口 + 前缀状态压缩
"""

from math import inf

class Solution:
    def maxDifference(self, s: str, k: int) -> int:
        s = list(map(int, s))
        ans = -inf
        for a in range(5):
            for b in range(5):
                if a == b:
                    continue
                curA = curB = 0
                preA = preB = 0
                t = [[inf, inf], [inf, inf]]
                l = -1
                for r, x in enumerate(s):
                    curA += x == a
                    curB += x == b
                    while r - l >= k and curB - preB >= 2:
                        t[preA & 1][preB & 1] = min(t[preA & 1][preB & 1], preA - preB)
                        l += 1
                        preA += s[l] == a
                        preB += s[l] == b
                    ans = max(ans, curA - curB - t[curA & 1 ^ 1][curB & 1])
        return ans

def _test():
    s = Solution()
    assert s.maxDifference(s='12233', k=4) == -1
    assert s.maxDifference(s='1122211', k=3) == 1
    assert s.maxDifference(s='110', k=3) == -1
    print("LC 3445 OK")

if __name__ == "__main__":
    _test()
