"""
LC 3209. Number of Subarrays With AND Value of K — Hard
https://leetcode.com/problems/number-of-subarrays-with-and-value-of-k/
Best: 方法一：哈希表 + 枚举
"""

from collections import Counter

class Solution:
    def countSubarrays(self, nums: list[int], k: int) -> int:
        ans = 0
        pre = Counter()
        for x in nums:
            cur = Counter()
            for y, v in pre.items():
                cur[x & y] += v
            cur[x] += 1
            ans += cur[k]
            pre = cur
        return ans

def _test():
    s = Solution()
    assert s.countSubarrays(nums=[1, 1, 1], k=1) == 6
    assert s.countSubarrays(nums=[1, 1, 2], k=1) == 3
    assert s.countSubarrays(nums=[1, 2, 3], k=2) == 2
    print("LC 3209 OK")

if __name__ == "__main__":
    _test()
