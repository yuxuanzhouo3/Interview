"""
LC 1220. Count Vowels Permutation — Hard
https://leetcode.com/problems/count-vowels-permutation/
Best: DP transitions on vowels — O(n)
"""


class Solution:
    def countVowelPermutation(self, n: int) -> int:
        MOD = 10**9 + 7
        # a e i o u
        a = e = i = o = u = 1
        for _ in range(n - 1):
            a, e, i, o, u = e, (a + i) % MOD, (a + e + o + u) % MOD, (i + u) % MOD, a
        return (a + e + i + o + u) % MOD


def _test():
    s = Solution()
    assert s.countVowelPermutation(1) == 5
    assert s.countVowelPermutation(2) == 10
    assert s.countVowelPermutation(5) == 68
    print("LC 1220 OK")


if __name__ == "__main__":
    _test()
