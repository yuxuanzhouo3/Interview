"""
LC 3321. Find X-Sum of All K-Long Subarrays II — Hard
https://leetcode.com/problems/find-x-sum-of-all-k-long-subarrays-ii/
Best: 方法一：哈希表 + 有序集合
"""

from collections import Counter
from sortedcontainers import SortedList

class Solution:
    def findXSum(self, nums: list[int], k: int, x: int) -> list[int]:
        def add(v: int):
            if cnt[v] == 0:
                return
            p = (cnt[v], v)
            if l and p > l[0]:
                nonlocal s
                s += p[0] * p[1]
                l.add(p)
            else:
                r.add(p)

        def remove(v: int):
            if cnt[v] == 0:
                return
            p = (cnt[v], v)
            if p in l:
                nonlocal s
                s -= p[0] * p[1]
                l.remove(p)
            else:
                r.remove(p)

        l = SortedList()
        r = SortedList()
        cnt = Counter()
        s = 0
        n = len(nums)
        ans = [0] * (n - k + 1)
        for i, v in enumerate(nums):
            remove(v)
            cnt[v] += 1
            add(v)
            j = i - k + 1
            if j < 0:
                continue
            while r and len(l) < x:
                p = r.pop()
                l.add(p)
                s += p[0] * p[1]
            while len(l) > x:
                p = l.pop(0)
                s -= p[0] * p[1]
                r.add(p)
            ans[j] = s

            remove(nums[j])
            cnt[nums[j]] -= 1
            add(nums[j])
        return ans

def _test():
    s = Solution()
    assert s.findXSum(nums=[1, 1, 2, 2, 3, 4, 2, 3], k=6, x=2) == [6, 10, 12]
    assert s.findXSum(nums=[3, 8, 7, 8, 7, 5], k=2, x=2) == [11, 15, 15, 15, 12]
    print("LC 3321 OK")

if __name__ == "__main__":
    _test()
