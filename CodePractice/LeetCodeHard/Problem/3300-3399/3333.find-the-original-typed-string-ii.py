"""
LC 3333. Find the Original Typed String II — Hard
https://leetcode.com/problems/find-the-original-typed-string-ii/
Best: 方法一：动态规划 + 前缀和
"""

from itertools import accumulate

class Solution:
    def possibleStringCount(self, word: str, k: int) -> int:
        mod = 10**9 + 7
        nums = []
        ans = 1
        cur = 0
        for i, c in enumerate(word):
            cur += 1
            if i == len(word) - 1 or c != word[i + 1]:
                if cur > 1:
                    if k > 0:
                        nums.append(cur - 1)
                    ans = ans * cur % mod
                cur = 0
                k -= 1
        if k < 1:
            return ans
        m = len(nums)
        f = [[0] * k for _ in range(m + 1)]
        f[0][0] = 1
        for i, x in enumerate(nums, 1):
            s = list(accumulate(f[i - 1], initial=0))
            for j in range(k):
                f[i][j] = (s[j + 1] - s[j - min(x, j)] + mod) % mod
        return (ans - sum(f[m][j] for j in range(k))) % mod

def _test():
    s = Solution()
    assert s.possibleStringCount(word='aabbccdd', k=7) == 5
    assert s.possibleStringCount(word='aabbccdd', k=8) == 1
    assert s.possibleStringCount(word='aaabbb', k=3) == 8
    print("LC 3333 OK")

if __name__ == "__main__":
    _test()
