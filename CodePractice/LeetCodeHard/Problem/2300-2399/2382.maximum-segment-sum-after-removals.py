"""
LC 2382. Maximum Segment Sum After Removals — Hard
https://leetcode.com/problems/maximum-segment-sum-after-removals/
Best: Reverse union-find add segments — O(n α(n))
"""


from typing import List


class Solution:
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        n = len(nums)
        parent = list(range(n))
        seg_sum = [0] * n
        active = [False] * n
        ans = [0] * n
        cur_max = 0

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            nonlocal cur_max
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            parent[rb] = ra
            seg_sum[ra] += seg_sum[rb]
            cur_max = max(cur_max, seg_sum[ra])

        for i in range(n - 1, -1, -1):
            ans[i] = cur_max
            idx = removeQueries[i]
            active[idx] = True
            seg_sum[idx] = nums[idx]
            cur_max = max(cur_max, nums[idx])
            if idx > 0 and active[idx - 1]:
                union(idx, idx - 1)
            if idx + 1 < n and active[idx + 1]:
                union(idx, idx + 1)
        return ans


def _test():
    s = Solution()
    assert s.maximumSegmentSum([1, 2, 5, 6, 1], [0, 3, 2, 4, 1]) == [14, 7, 2, 2, 0]
    assert s.maximumSegmentSum([3, 2, 11, 1], [3, 2, 1, 0]) == [16, 5, 3, 0]
    print("LC 2382 OK")


if __name__ == "__main__":
    _test()

