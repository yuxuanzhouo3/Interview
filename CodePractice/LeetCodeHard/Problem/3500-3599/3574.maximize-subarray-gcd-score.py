"""
LC 3574. Maximize Subarray GCD Score — Hard
https://leetcode.com/problems/maximize-subarray-gcd-score/
Best: 方法一：枚举 + 数学
"""

from math import inf, gcd

class Solution:
    def maxGCDScore(self, nums: list[int], k: int) -> int:
        n = len(nums)
        cnt = [0] * n
        for i, x in enumerate(nums):
            while x % 2 == 0:
                cnt[i] += 1
                x //= 2
        ans = 0
        for l in range(n):
            g = 0
            mi = inf
            t = 0
            for r in range(l, n):
                g = gcd(g, nums[r])
                if cnt[r] < mi:
                    mi = cnt[r]
                    t = 1
                elif cnt[r] == mi:
                    t += 1
                ans = max(ans, (g if t > k else g * 2) * (r - l + 1))
        return ans

def _test():
    s = Solution()
    assert s.maxGCDScore(nums=[2, 4], k=1) == 8
    assert s.maxGCDScore(nums=[3, 5, 7], k=2) == 14
    assert s.maxGCDScore(nums=[5, 5, 5], k=1) == 15
    print("LC 3574 OK")

if __name__ == "__main__":
    _test()
