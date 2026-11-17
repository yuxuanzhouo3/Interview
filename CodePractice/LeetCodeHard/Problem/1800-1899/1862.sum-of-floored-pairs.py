"""
LC 1862. Sum of Floored Pairs — Hard
https://leetcode.com/problems/sum-of-floored-pairs/
Best: Counting + prefix floor sum enum — O(M log M)
"""


from typing import List


class Solution:
    def sumOfFlooredPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        M = max(nums)
        cnt = [0] * (M + 1)
        for x in nums:
            cnt[x] += 1
        pref = [0] * (M + 1)
        for i in range(1, M + 1):
            pref[i] = pref[i - 1] + cnt[i]
        ans = 0
        for y in range(1, M + 1):
            if cnt[y] == 0:
                continue
            # for multiples of y: floor(x/y)=k for x in [k*y, (k+1)*y)
            k = 1
            while k * y <= M:
                L = k * y
                R = min(M, (k + 1) * y - 1)
                ans += cnt[y] * k * (pref[R] - pref[L - 1])
                ans %= MOD
                k += 1
        return ans


def _test():
    s = Solution()
    assert s.sumOfFlooredPairs([2, 5, 9]) == 10
    assert s.sumOfFlooredPairs([7, 7, 7, 7, 7, 7, 7]) == 49
    print("LC 1862 OK")


if __name__ == "__main__":
    _test()

