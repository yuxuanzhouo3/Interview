"""
LC 3464. Maximize the Distance Between Points on a Square — Hard
https://leetcode.com/problems/maximize-the-distance-between-points-on-a-square/
Best: 方法一：二分答案 + 坐标映射 + 贪心
"""

from bisect import bisect_left

class Solution:
    def maxDistance(self, side: int, points: list[list[int]], k: int) -> int:
        nums = []
        for x, y in points:
            if x == 0:
                nums.append(y)
            elif y == side:
                nums.append(side + x)
            elif x == side:
                nums.append(side * 3 - y)
            else:
                nums.append(side * 4 - x)
        nums.sort()

        def check(lo: int) -> bool:
            for start in nums:
                end = start + side * 4 - lo
                cur = start
                ok = True
                for _ in range(k - 1):
                    j = bisect_left(nums, cur + lo)
                    if j == len(nums) or nums[j] > end:
                        ok = False
                        break
                    cur = nums[j]
                if ok:
                    return True
            return False

        l, r = 1, side
        while l < r:
            mid = (l + r + 1) >> 1
            if check(mid):
                l = mid
            else:
                r = mid - 1
        return l

def _test():
    s = Solution()
    assert s.maxDistance(side=2, points=[[0, 2], [2, 0], [2, 2], [0, 0]], k=4) == 2
    assert s.maxDistance(side=2, points=[[0, 0], [1, 2], [2, 0], [2, 2], [2, 1]], k=4) == 1
    assert s.maxDistance(side=2, points=[[0, 0], [0, 1], [0, 2], [1, 2], [2, 0], [2, 2], [2, 1]], k=5) == 1
    print("LC 3464 OK")

if __name__ == "__main__":
    _test()
