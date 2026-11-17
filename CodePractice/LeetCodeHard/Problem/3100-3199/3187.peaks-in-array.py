"""
LC 3187. Peaks in Array — Hard
https://leetcode.com/problems/peaks-in-array/
Best: BIT maintain peaks — O((n+q) log n)
"""

from typing import List


class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        bit = [0] * (n + 2)

        def add(i, d):
            i += 1
            while i <= n:
                bit[i] += d
                i += i & -i

        def sum_(i):
            if i < 0:
                return 0
            i += 1
            s = 0
            while i > 0:
                s += bit[i]
                i -= i & -i
            return s

        def is_peak(i):
            return 0 < i < n - 1 and nums[i] > nums[i - 1] and nums[i] > nums[i + 1]

        peak = [False] * n
        for i in range(1, n - 1):
            if is_peak(i):
                peak[i] = True
                add(i, 1)

        def upd_peak(i):
            if not (0 < i < n - 1):
                return
            now = is_peak(i)
            if now and not peak[i]:
                peak[i] = True
                add(i, 1)
            elif not now and peak[i]:
                peak[i] = False
                add(i, -1)

        ans = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                if r - l + 1 < 3:
                    ans.append(0)
                else:
                    # peaks with index in (l, r)
                    ans.append(sum_(r - 1) - sum_(l))
            else:
                idx, val = q[1], q[2]
                nums[idx] = val
                for i in (idx - 1, idx, idx + 1):
                    upd_peak(i)
        return ans


def _test():
    s = Solution()
    assert s.countOfPeaks([3, 1, 4, 2, 5], [[2, 3, 4], [1, 0, 4]]) == [0]
    # LC ex1: nums = [3,1,4,2,5], queries = [[2,3,4],[1,0,4]]
    # After update nums[3]=4 -> [3,1,4,4,5]; peaks in (0,4): none -> [0]
    assert s.countOfPeaks([4, 1, 4, 2, 1, 5], [[2, 2, 4], [1, 0, 2], [1, 0, 4]]) == [0, 1]
    print("LC 3187 OK")


if __name__ == "__main__":
    _test()

