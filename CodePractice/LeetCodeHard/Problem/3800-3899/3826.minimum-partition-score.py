"""
LC 3826. Minimum Partition Score — Hard
https://leetcode.com/problems/minimum-partition-score/
Best: DP partition with convex cost; Li Chao / CHT optimized (here O(n^2 k) verified)
"""
INF = 10**18


class Solution:
    def minPartitionScore(self, nums: list[int], k: int) -> int:
        n = len(nums)
        P = [0] * (n + 1)
        for i, x in enumerate(nums):
            P[i + 1] = P[i] + x

        def tri(s: int) -> int:
            return s * (s + 1) // 2

        dp = [INF] * (n + 1)
        dp[0] = 0
        for _ in range(k):
            ndp = [INF] * (n + 1)
            for i in range(1, n + 1):
                best = INF
                for j in range(i):
                    if dp[j] >= INF:
                        continue
                    s = P[i] - P[j]
                    best = min(best, dp[j] + tri(s))
                ndp[i] = best
            dp = ndp
        return dp[n]


def _test():
    s = Solution()
    assert s.minPartitionScore([5, 1, 2, 1], 2) == 25
    assert s.minPartitionScore([1, 2, 3, 4], 1) == 55
    assert s.minPartitionScore([1, 1, 1], 3) == 3
    print("LC 3826 OK")


if __name__ == "__main__":
    _test()
