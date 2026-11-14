"""
LC 552. Student Attendance Record II — Hard
https://leetcode.com/problems/student-attendance-record-ii/
Best: DP states (absences, consecutive L) — O(n)
"""

MOD = 10**9 + 7


class Solution:
    def checkRecord(self, n: int) -> int:
        # dp[a][l]: ways with a absences used, ending with l consecutive Lates
        dp = [[0] * 3 for _ in range(2)]
        dp[0][0] = 1
        for _ in range(n):
            ndp = [[0] * 3 for _ in range(2)]
            for a in range(2):
                for l in range(3):
                    v = dp[a][l]
                    if not v:
                        continue
                    # P
                    ndp[a][0] = (ndp[a][0] + v) % MOD
                    # A
                    if a < 1:
                        ndp[a + 1][0] = (ndp[a + 1][0] + v) % MOD
                    # L
                    if l < 2:
                        ndp[a][l + 1] = (ndp[a][l + 1] + v) % MOD
            dp = ndp
        return sum(dp[a][l] for a in range(2) for l in range(3)) % MOD


def _test():
    s = Solution()
    assert s.checkRecord(1) == 3
    assert s.checkRecord(2) == 8
    assert s.checkRecord(10101) == 183236316
    print("LC 552 OK")


if __name__ == "__main__":
    _test()
