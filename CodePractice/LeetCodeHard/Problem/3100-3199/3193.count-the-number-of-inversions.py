"""
LC 3193. Count the Number of Inversions — Hard
https://leetcode.com/problems/count-the-number-of-inversions/
Best: DP on prefix inversions with constraints — O(n * cnt * maxreq)
"""

from typing import List


class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        req = [-1] * n
        for end, cnt in requirements:
            req[end] = cnt
        if req[0] > 0:
            return 0
        if req[0] == -1:
            req[0] = 0
        maxc = max(c for _, c in requirements)
        # dp[i][j] = ways to place 0..i with j inversions (relative ranks)
        dp = [0] * (maxc + 1)
        dp[0] = 1
        for i in range(1, n):
            ndp = [0] * (maxc + 1)
            pref = [0] * (maxc + 2)
            for j in range(maxc + 1):
                pref[j + 1] = (pref[j] + dp[j]) % MOD
            for j in range(maxc + 1):
                # insert new largest at position creating t inversions, t=0..i, j-t>=0
                lo = j - i
                if lo < 0:
                    lo = 0
                ndp[j] = (pref[j + 1] - pref[lo]) % MOD
            if req[i] != -1:
                for j in range(maxc + 1):
                    if j != req[i]:
                        ndp[j] = 0
            dp = ndp
        return sum(dp) % MOD if req[n - 1] == -1 else dp[req[n - 1]] % MOD


def _test():
    s = Solution()
    assert s.numberOfPermutations(3, [[2, 2], [0, 0]]) == 2
    assert s.numberOfPermutations(3, [[2, 2], [1, 1], [0, 0]]) == 1
    assert s.numberOfPermutations(2, [[0, 0], [1, 0]]) == 1
    print("LC 3193 OK")


if __name__ == "__main__":
    _test()

