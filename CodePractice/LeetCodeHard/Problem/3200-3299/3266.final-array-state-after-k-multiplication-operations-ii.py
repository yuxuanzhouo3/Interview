"""
LC 3266. Final Array State After K Multiplication Operations II — Hard
https://leetcode.com/problems/final-array-state-after-k-multiplication-operations-ii/
Best: 方法一：优先队列（小根堆）+ 模拟
"""

from heapq import heappush, heappop, heapify

class Solution:
    def getFinalState(self, nums: list[int], k: int, multiplier: int) -> list[int]:
        if multiplier == 1:
            return nums
        pq = [(x, i) for i, x in enumerate(nums)]
        heapify(pq)
        m = max(nums)
        while k and pq[0][0] < m:
            x, i = heappop(pq)
            heappush(pq, (x * multiplier, i))
            k -= 1
        n = len(nums)
        mod = 10**9 + 7
        pq.sort()
        for i, (x, j) in enumerate(pq):
            nums[j] = x * pow(multiplier, k // n + int(i < k % n), mod) % mod
        return nums

def _test():
    s = Solution()
    assert s.getFinalState(nums=[2, 1, 3, 5, 6], k=5, multiplier=2) == [8, 4, 6, 5, 6]
    assert s.getFinalState(nums=[100000, 2000], k=2, multiplier=1000000) == [999999307, 999999993]
    print("LC 3266 OK")

if __name__ == "__main__":
    _test()
