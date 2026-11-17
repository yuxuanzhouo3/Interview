"""
LC 3957. Maximum Sum of M Non-Overlapping Subarrays II — Hard
https://leetcode.com/problems/maximum-sum-of-m-non-overlapping-subarrays-ii/
Best: same as I for shared samples — DP + Lagrangian on segment count
"""
from collections import deque


class Solution:
    def maximumSum(self, nums: list[int], m: int, l: int, r: int) -> int:
        n = len(nums)
        pre = [0] * (n + 1)
        for i, x in enumerate(nums):
            pre[i + 1] = pre[i] + x

        best_one = -(10**18)
        for length in range(l, r + 1):
            for i in range(0, n - length + 1):
                best_one = max(best_one, pre[i + length] - pre[i])

        def helper(cost: int):
            dp = [0] * (n + 1)
            cnt = [0] * (n + 1)
            dq = deque()

            def val(i):
                return dp[i] - pre[i]

            for i in range(l, n + 1):
                add = i - l
                while dq and (val(dq[-1]) < val(add) or (val(dq[-1]) == val(add) and cnt[dq[-1]] <= cnt[add])):
                    dq.pop()
                dq.append(add)
                while dq and dq[0] < i - r:
                    dq.popleft()
                dp[i] = dp[i - 1]
                cnt[i] = cnt[i - 1]
                if dq:
                    take = val(dq[0]) + pre[i] - cost
                    take_cnt = cnt[dq[0]] + 1
                    if take > dp[i] or (take == dp[i] and take_cnt > cnt[i]):
                        dp[i] = take
                        cnt[i] = take_cnt
            return dp[n], cnt[n]

        zero_v, zero_c = helper(0)
        if zero_c == 0:
            return best_one
        if zero_c <= m:
            return zero_v
        lo, hi = 0, 10**18
        while lo <= hi:
            mid = (lo + hi) // 2
            v, c = helper(mid)
            if c > m:
                lo = mid + 1
            else:
                hi = mid - 1
        v, c = helper(hi)
        return v + hi * m

def _test():
    s = Solution()
    assert s.maximumSum(nums=[4, 1, -5, 2], m=2, l=1, r=3) == 7
    assert s.maximumSum(nums=[1, 0, 3, 4], m=2, l=1, r=2) == 8
    assert s.maximumSum(nums=[-1, 7, -4], m=1, l=2, r=3) == 6
    assert s.maximumSum(nums=[-3, -4, -1], m=2, l=1, r=2) == -1
    print("LC 3957 OK")


if __name__ == "__main__":
    _test()
