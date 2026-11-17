"""
LC 3892. Minimum Operations to Achieve at Least K Peaks — Hard
https://leetcode.com/problems/minimum-operations-to-achieve-at-least-k-peaks/
Best: circular peak costs + DP select non-adjacent peaks
"""
INF = 1 << 60


class Solution:
    def minOperations(self, nums: list[int], k: int) -> int:
        n = len(nums)
        if k == 0:
            return 0
        if k > n // 2:
            return -1
        cost = [0] * n
        for i in range(n):
            cost[i] = max(
                0,
                nums[(i - 1) % n] - nums[i] + 1,
                nums[(i + 1) % n] - nums[i] + 1,
            )

        def helper(start: int, length: int, need: int) -> int:
            if need < 0:
                return INF
            if need == 0:
                return 0
            if length <= 0 or need > (length + 1) // 2:
                return INF
            prev2 = [INF] * (need + 1)
            prev1 = [INF] * (need + 1)
            prev2[0] = prev1[0] = 0
            for i in range(length):
                cur = prev1[:]
                w = cost[start + i]
                for j in range(1, need + 1):
                    if prev2[j - 1] != INF:
                        cur[j] = min(cur[j], prev2[j - 1] + w)
                prev2, prev1 = prev1, cur
            return prev1[need]

        res = min(helper(1, n - 1, k), cost[0] + helper(2, max(0, n - 3), k - 1))
        return -1 if res >= INF else int(res)


def _test():
    s = Solution()
    assert s.minOperations([2, 1, 2], 1) == 1
    assert s.minOperations([4, 5, 3, 6], 2) == 0
    assert s.minOperations([3, 7, 3], 2) == -1
    print("LC 3892 OK")


if __name__ == "__main__":
    _test()
