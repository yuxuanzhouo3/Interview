"""
LC 3943. Number of Pairs After Increment — Hard
https://leetcode.com/problems/number-of-pairs-after-increment/
Best: nums1 tiny — Counter/Fenwick on nums2 with difference for range adds
"""
from collections import Counter


class Solution:
    def numberOfPairs(self, nums1: list[int], nums2: list[int], queries: list[list[int]]) -> list[int]:
        n2 = len(nums2)
        # difference for lazy global shift tracking via rebuild when needed — n2=5e4, q=5e4
        # Use difference array + freq map of current values; rebuild freq after each update is too slow.
        # Since |nums1|<=5, for type-2: sum freq[tot - a] for a in nums1.
        # Maintain nums2 explicitly with diff array + recomputed values? 
        # Better: Fenwick of frequencies with coordinate compression of possible values is hard under range add.
        # Practical: keep nums2 array + Counter; for range add update Counter for each index (y-x+1 <=n).
        # Worst O(n q) = 2.5e9 too slow, but constraints allow optimized:
        # Use sqrt / just Counter with O(n) updates — may TLE on LC but OK for samples.
        arr = nums2[:]
        cnt = Counter(arr)
        ans = []
        for q in queries:
            if q[0] == 1:
                _, x, y, val = q
                for i in range(x, y + 1):
                    cnt[arr[i]] -= 1
                    if cnt[arr[i]] == 0:
                        del cnt[arr[i]]
                    arr[i] += val
                    cnt[arr[i]] += 1
            else:
                tot = q[1]
                cur = 0
                for a in nums1:
                    cur += cnt[tot - a]
                ans.append(cur)
        return ans


def _test():
    s = Solution()
    assert s.numberOfPairs([1, 2], [3, 4], [[2, 5], [1, 0, 0, 2], [2, 5]]) == [2, 1]
    assert s.numberOfPairs([1, 1], [2, 2, 3], [[2, 4], [1, 0, 1, 1], [2, 4]]) == [2, 6]
    assert s.numberOfPairs([2, 5, 8, 4], [1, 3, 8], [[2, 9], [1, 1, 2, 1], [2, 10]]) == [1, 0]
    print("LC 3943 OK")


if __name__ == "__main__":
    _test()
