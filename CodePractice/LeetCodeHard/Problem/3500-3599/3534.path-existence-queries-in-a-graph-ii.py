"""
LC 3534. Path Existence Queries in a Graph II — Hard
https://leetcode.com/problems/path-existence-queries-in-a-graph-ii/
Best: 方法一：排序 + 二分倍增
"""

class Solution:
    def pathExistenceQueries(
        self, n: int, nums: list[int], maxDiff: int, queries: list[list[int]]
    ) -> list[int]:
        pairs = sorted((x, i) for i, x in enumerate(nums))
        m = 20
        f = [[0] * m for _ in range(n)]
        r = n - 1
        for l in range(n - 1, -1, -1):
            while pairs[r][0] - pairs[l][0] > maxDiff:
                r -= 1
            i, j = pairs[l][1], pairs[r][1]
            f[i][0] = j
            for k in range(1, m):
                f[i][k] = f[f[i][k - 1]][k - 1]

        ans = []
        for i, j in queries:
            if nums[i] > nums[j]:
                i, j = j, i
            if i == j:
                ans.append(0)
                continue
            if nums[i] == nums[j]:
                ans.append(1)
                continue
            d = 0
            for k in range(m - 1, -1, -1):
                if nums[f[i][k]] < nums[j]:
                    d |= 1 << k
                    i = f[i][k]
            if nums[f[i][0]] < nums[j]:
                ans.append(-1)
            else:
                ans.append(d + 1)
        return ans

def _test():
    s = Solution()
    assert s.pathExistenceQueries(n=5, nums=[1, 8, 3, 4, 2], maxDiff=3, queries=[[0, 3], [2, 4]]) == [1, 1]
    assert s.pathExistenceQueries(n=5, nums=[5, 3, 1, 9, 10], maxDiff=2, queries=[[0, 1], [0, 2], [2, 3], [4, 3]]) == [1, 2, -1, 1]
    assert s.pathExistenceQueries(n=3, nums=[3, 6, 1], maxDiff=1, queries=[[0, 0], [0, 1], [1, 2]]) == [0, -1, -1]
    print("LC 3534 OK")

if __name__ == "__main__":
    _test()
