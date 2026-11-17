"""
LC 1830. Minimum Number of Operations to Make String Sorted — Hard
https://leetcode.com/problems/minimum-number-of-operations-to-make-string-sorted/
Best: Count smaller permutations — O(n·26)
"""

from collections import Counter

MOD = 10**9 + 7
N = 3010
fact = [1] * N
invf = [1] * N
for i in range(1, N):
    fact[i] = fact[i - 1] * i % MOD
    invf[i] = pow(fact[i], MOD - 2, MOD)

class Solution:
    def makeStringSorted(self, s: str) -> int:
        cnt = Counter(s)
        ans, n = 0, len(s)
        for i, c in enumerate(s):
            m = sum(v for a, v in cnt.items() if a < c)
            t = fact[n - i - 1] * m % MOD
            for v in cnt.values():
                t = t * invf[v] % MOD
            ans = (ans + t) % MOD
            cnt[c] -= 1
            if cnt[c] == 0:
                del cnt[c]
        return ans

def _test():

    s = Solution()
    assert s.makeStringSorted("cba") == 5
    assert s.makeStringSorted("aabaa") == 2
    assert s.makeStringSorted("cdbea") == 63
    assert s.makeStringSorted("leetcodeleetcodeleetcode") == 982157772

    print("LC 1830 OK")

if __name__ == "__main__":
    _test()
