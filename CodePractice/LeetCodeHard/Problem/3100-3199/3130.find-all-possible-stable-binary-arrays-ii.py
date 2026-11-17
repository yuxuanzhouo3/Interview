"""
LC 3130. Find All Possible Stable Binary Arrays II — Hard
https://leetcode.com/problems/find-all-possible-stable-binary-arrays-ii/
Best: DP by last block — O(zero * one * limit)
"""

MOD = 10**9 + 7


class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        dp0 = [[0] * (one + 1) for _ in range(zero + 1)]
        dp1 = [[0] * (one + 1) for _ in range(zero + 1)]
        for i in range(1, min(zero, limit) + 1):
            dp0[i][0] = 1
        for j in range(1, min(one, limit) + 1):
            dp1[0][j] = 1
        for i in range(zero + 1):
            for j in range(one + 1):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j > 0:
                    for k in range(1, limit + 1):
                        if i - k >= 0:
                            dp0[i][j] = (dp0[i][j] + dp1[i - k][j]) % MOD
                        if j - k >= 0:
                            dp1[i][j] = (dp1[i][j] + dp0[i][j - k]) % MOD
        return (dp0[zero][one] + dp1[zero][one]) % MOD


def _test():
    s = Solution()
    assert s.numberOfStableArrays(1, 1, 2) == 2
    assert s.numberOfStableArrays(1, 2, 1) == 1
    assert s.numberOfStableArrays(3, 3, 2) == 14
    print("LC 3130 OK")


if __name__ == "__main__":
    _test()
