"""
LC 3395. Subsequences with a Unique Middle Mode I — Hard
https://leetcode.com/problems/subsequences-with-a-unique-middle-mode-i/
Best: Fix middle + subtract non-unique modes — O(n |Sigma|)
"""

import collections
from typing import List


class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ans = 0
        p = collections.Counter()
        s = collections.Counter(nums)

        def nC2(n: int) -> int:
            return n * (n - 1) // 2

        for i, a in enumerate(nums):
            s[a] -= 1
            l = i
            r = len(nums) - i - 1
            ans += nC2(l) * nC2(r)
            ans -= nC2(l - p[a]) * nC2(r - s[a])
            for b in set(p) | set(s):
                if b == a:
                    continue
                ans -= p[a] * p[b] * s[b] * (r - s[a] - s[b])
                ans -= s[a] * s[b] * p[b] * (l - p[a] - p[b])
                ans -= nC2(p[b]) * s[a] * (r - s[a] - s[b])
                ans -= nC2(s[b]) * p[a] * (l - p[a] - p[b])
                ans -= nC2(p[b]) * s[a] * s[b]
                ans -= nC2(s[b]) * p[a] * p[b]
            ans %= MOD
            p[a] += 1
        return ans % MOD


def _test():
    s = Solution()
    assert s.subsequencesWithMiddleMode([1, 1, 1, 1, 1, 1]) == 6
    assert s.subsequencesWithMiddleMode([1, 2, 2, 3, 3, 4]) == 4
    assert s.subsequencesWithMiddleMode([0, 1, 2, 3, 4, 5, 6, 7, 8]) == 0
    print("LC 3395 OK")


if __name__ == "__main__":
    _test()
