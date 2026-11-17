"""
LC 2954. Count the Number of Infection Sequences — Hard
https://leetcode.com/problems/count-the-number-of-infection-sequences/
Best: Combinatorics gaps * 2^{len-1} — O(n)
"""

from typing import List


class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        # factorial
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i % MOD
        invfact = [1] * (n + 1)
        invfact[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n, 0, -1):
            invfact[i - 1] = invfact[i] * i % MOD

        def C(a, b):
            if b < 0 or b > a:
                return 0
            return fact[a] * invfact[b] % MOD * invfact[a - b] % MOD

        total_empty = n - len(sick)
        ans = fact[total_empty]
        # divide by factorials of gap sizes (multinomial), then multiply middle gap 2^{len-1}
        gaps = []
        # left gap
        left = sick[0] - 0
        gaps.append(('end', left))
        for i in range(1, len(sick)):
            gaps.append(('mid', sick[i] - sick[i - 1] - 1))
        gaps.append(('end', n - 1 - sick[-1]))
        for typ, g in gaps:
            if g == 0:
                continue
            ans = ans * invfact[g] % MOD
            if typ == 'mid' and g >= 1:
                ans = ans * pow(2, g - 1, MOD) % MOD
        return ans


def _test():
    s = Solution()
    assert s.numberOfSequence(5, [0, 4]) == 4
    assert s.numberOfSequence(4, [1]) == 3
    print("LC 2954 OK")


if __name__ == "__main__":
    _test()
