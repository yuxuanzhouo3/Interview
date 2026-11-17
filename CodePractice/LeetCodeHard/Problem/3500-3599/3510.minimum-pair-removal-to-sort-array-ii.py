"""
LC 3510. Minimum Pair Removal to Sort Array II — Hard
https://leetcode.com/problems/minimum-pair-removal-to-sort-array-ii/
Best: 方法一：有序集合
"""

from sortedcontainers import SortedList

class Solution:
    def minimumPairRemoval(self, nums: list[int]) -> int:
        n = len(nums)
        sl = SortedList()
        idx = SortedList(range(n))
        inv = 0
        for i in range(n - 1):
            sl.add((nums[i] + nums[i + 1], i))
            if nums[i] > nums[i + 1]:
                inv += 1
        ans = 0
        while inv:
            ans += 1
            s, i = sl.pop(0)
            pos = idx.index(i)
            j = idx[pos + 1]
            if nums[i] > nums[j]:
                inv -= 1
            if pos > 0:
                h = idx[pos - 1]
                if nums[h] > nums[i]:
                    inv -= 1
                sl.remove((nums[h] + nums[i], h))
                if nums[h] > s:
                    inv += 1
                sl.add((nums[h] + s, h))
            if pos + 2 < len(idx):
                k = idx[pos + 2]
                if nums[j] > nums[k]:
                    inv -= 1
                sl.remove((nums[j] + nums[k], j))
                if s > nums[k]:
                    inv += 1
                sl.add((s + nums[k], i))

            nums[i] = s
            idx.remove(j)
        return ans

def _test():
    s = Solution()
    assert s.minimumPairRemoval(nums=[5, 2, 3, 1]) == 2
    assert s.minimumPairRemoval(nums=[1, 2, 2]) == 0
    print("LC 3510 OK")

if __name__ == "__main__":
    _test()
