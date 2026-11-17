"""
LC 3337. Total Characters in String After Transformations II — Hard
https://leetcode.com/problems/total-characters-in-string-after-transformations-ii/
Best: Matrix exponentiation 26x26 — O(26^3 log t)
"""

from typing import List


MOD = 10**9 + 7


class Solution:
    def lengthAfterTransformations(self, s: str, t: int, nums: List[int]) -> int:
        # M[j][i] = 1 if char i transforms to include j (actually: next[c] += cur[prev] for each)
        # After one step: for char c, it becomes (c+1)%26, (c+2)%26, ..., (c+nums[c])%26 each once
        M = [[0] * 26 for _ in range(26)]
        for c in range(26):
            for k in range(1, nums[c] + 1):
                M[(c + k) % 26][c] += 1

        def mul(A, B):
            C = [[0] * 26 for _ in range(26)]
            for i in range(26):
                for k in range(26):
                    if A[i][k] == 0:
                        continue
                    for j in range(26):
                        C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % MOD
            return C

        def power(A, e):
            R = [[1 if i == j else 0 for j in range(26)] for i in range(26)]
            while e:
                if e & 1:
                    R = mul(A, R)
                A = mul(A, A)
                e >>= 1
            return R

        Mt = power(M, t)
        cnt = [0] * 26
        for ch in s:
            cnt[ord(ch) - 97] += 1
        ans = 0
        for i in range(26):
            for j in range(26):
                ans = (ans + Mt[i][j] * cnt[j]) % MOD
        return ans


def _test():
    s = Solution()
    assert (
        s.lengthAfterTransformations(
            "abcyy",
            2,
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
        )
        == 7
    )
    assert (
        s.lengthAfterTransformations(
            "azbk",
            1,
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
        == 8
    )
    print("LC 3337 OK")


if __name__ == "__main__":
    _test()
