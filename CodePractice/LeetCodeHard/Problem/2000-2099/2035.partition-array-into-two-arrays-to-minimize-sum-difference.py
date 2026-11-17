"""
LC 2035. Partition Array Into Two Arrays to Minimize Sum Difference — Hard
https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/
Best: Meet-in-the-middle subset sums — O(2^{n/2} n)
"""


from typing import List
import bisect


class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        n = len(nums) // 2
        total = sum(nums)

        def subset_sums(arr):
            m = len(arr)
            res = [[] for _ in range(m + 1)]
            for mask in range(1 << m):
                s = cnt = 0
                for i in range(m):
                    if mask & (1 << i):
                        s += arr[i]
                        cnt += 1
                res[cnt].append(s)
            for i in range(m + 1):
                res[i].sort()
            return res

        left = subset_sums(nums[:n])
        right = subset_sums(nums[n:])
        ans = abs(sum(nums[:n]) - sum(nums[n:]))
        # pick k from left, n-k from right; sum_part close to total/2
        for k in range(n + 1):
            R = right[n - k]
            for s in left[k]:
                # want s + t ≈ total/2
                target = total // 2 - s
                j = bisect.bisect_left(R, target)
                for jj in (j - 1, j):
                    if 0 <= jj < len(R):
                        part = s + R[jj]
                        ans = min(ans, abs(total - 2 * part))
        return ans


def _test():
    s = Solution()
    assert s.minimumDifference([3, 9, 7, 3]) == 2
    assert s.minimumDifference([-36, 36]) == 72
    assert s.minimumDifference([2, -1, 0, 4, -2, -9]) == 0
    print("LC 2035 OK")


if __name__ == "__main__":
    _test()

