"""
LC 2478. Number of Beautiful Partitions — Hard
https://leetcode.com/problems/number-of-beautiful-partitions/
Best: DP prime-start/end partitions — O(n*k)
"""


class Solution:
    def beautifulPartitions(self, s: str, k: int, minLength: int) -> int:
        MOD = 10**9 + 7
        primes = set("2357")
        n = len(s)
        if s[0] not in primes or s[-1] in primes:
            return 0

        def can_start(i):
            return i < n and s[i] in primes

        def can_end(i):
            return i < n and s[i] not in primes

        # dp[i][p] ways to partition s[i:] into p parts
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[n][0] = 1
        for i in range(n - 1, -1, -1):
            for p in range(1, k + 1):
                if not can_start(i):
                    dp[i][p] = 0
                    continue
                # first part ends at j, length >= minLength, j endable, next startable or done
                for j in range(i + minLength - 1, n):
                    if not can_end(j):
                        continue
                    if j + 1 == n:
                        dp[i][p] = (dp[i][p] + dp[n][p - 1]) % MOD
                    elif can_start(j + 1):
                        dp[i][p] = (dp[i][p] + dp[j + 1][p - 1]) % MOD
        return dp[0][k]


def _test():
    s = Solution()
    assert s.beautifulPartitions("23542185131", 3, 2) == 3
    assert s.beautifulPartitions("23542185131", 3, 3) == 1
    assert s.beautifulPartitions("3312958", 3, 1) == 1
    print("LC 2478 OK")


if __name__ == "__main__":
    _test()

