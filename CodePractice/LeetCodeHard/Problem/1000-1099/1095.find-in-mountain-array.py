"""
LC 1095. Find in Mountain Array — Hard
https://leetcode.com/problems/find-in-mountain-array/
Best: 3 binary searches — O(log n)
"""


class MountainArray:
    def __init__(self, arr):
        self.arr = arr
        self.calls = 0

    def get(self, index: int) -> int:
        self.calls += 1
        return self.arr[index]

    def length(self) -> int:
        return len(self.arr)


class Solution:
    def findInMountainArray(self, target: int, mountain_arr: "MountainArray") -> int:
        n = mountain_arr.length()
        # find peak
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                lo = mid + 1
            else:
                hi = mid
        peak = lo
        # left ascending
        lo, hi = 0, peak
        while lo <= hi:
            mid = (lo + hi) // 2
            val = mountain_arr.get(mid)
            if val == target:
                return mid
            if val < target:
                lo = mid + 1
            else:
                hi = mid - 1
        # right descending
        lo, hi = peak + 1, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            val = mountain_arr.get(mid)
            if val == target:
                return mid
            if val > target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1


def _test():
    s = Solution()
    m = MountainArray([1, 2, 3, 4, 5, 3, 1])
    assert s.findInMountainArray(3, m) == 2
    m2 = MountainArray([0, 1, 2, 4, 2, 1])
    assert s.findInMountainArray(3, m2) == -1
    m3 = MountainArray([1, 5, 2])
    assert s.findInMountainArray(2, m3) == 2
    print("LC 1095 OK")


if __name__ == "__main__":
    _test()
