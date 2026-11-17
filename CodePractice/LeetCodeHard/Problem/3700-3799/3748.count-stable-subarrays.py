from bisect import bisect_left, bisect_right
"""
LC 3748. Count Stable Subarrays — Hard
https://leetcode.com/problems/count-stable-subarrays/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def countStableSubarrays(
        self, nums: list[int], queries: list[list[int]]
    ) -> list[int]:
        s = [0]
        l, n = 0, len(nums)
        seg = []
        for r, x in enumerate(nums):
            if r == n - 1 or x > nums[r + 1]:
                seg.append(l)
                k = r - l + 1
                s.append(s[-1] + (1 + k) * k // 2)
                l = r + 1
        ans = []
        for l, r in queries:
            i = bisect_right(seg, l)
            j = bisect_right(seg, r) - 1
            if i > j:
                k = r - l + 1
                ans.append((1 + k) * k // 2)
            else:
                a = seg[i] - l
                b = r - seg[j] + 1
                ans.append((1 + a) * a // 2 + s[j] - s[i] + (1 + b) * b // 2)
        return ans

def _test():
    s = Solution()
    assert s.countStableSubarrays([3,1,2], [[0,1],[1,2],[0,2]]) == [2,3,4]
    assert s.countStableSubarrays([2,2], [[0,1],[0,0]]) == [3,1]
    print("LC 3748 OK")


if __name__ == "__main__":
    _test()
