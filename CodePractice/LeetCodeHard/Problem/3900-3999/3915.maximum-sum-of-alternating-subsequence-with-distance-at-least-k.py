"""
LC 3915. Maximum Sum of Alternating Subsequence With Distance at Least K — Hard
https://leetcode.com/problems/maximum-sum-of-alternating-subsequence-with-distance-at-least-k/
Best: doocs/leetcode reference solution (verified on examples)
"""


class BinaryIndexedTree:
    def __init__(self, n: int):
        self.n = n
        self.c = [0] * (n + 1)

    def update(self, x: int, val: int) -> None:
        while x <= self.n:
            self.c[x] = max(self.c[x], val)
            x += x & -x

    def query(self, x: int) -> int:
        ans = 0
        while x > 0:
            ans = max(ans, self.c[x])
            x -= x & -x
        return ans


class Solution:
    def maxAlternatingSum(self, nums: list[int], k: int) -> int:
        vals = sorted(set(nums))
        m = len(vals)
        rank = {v: i + 1 for i, v in enumerate(vals)}
        bit0 = BinaryIndexedTree(m)
        bit1 = BinaryIndexedTree(m)
        n = len(nums)
        f = [[0, 0] for _ in range(n)]
        ans = 0
        for i, x in enumerate(nums):
            if i >= k:
                r = rank[nums[i - k]]
                bit0.update(r, f[i - k][0])
                bit1.update(m + 1 - r, f[i - k][1])
            r = rank[x]
            f[i][0] = x + bit1.query(m - r)
            f[i][1] = x + bit0.query(r - 1)
            ans = max(ans, f[i][0], f[i][1])
        return ans

def _test():
    s = Solution()
    assert s.maxAlternatingSum([5,4,2], 2) == 7
    assert s.maxAlternatingSum([3,5,4,2,4], 1) == 14
    assert s.maxAlternatingSum([5], 1) == 5
    print("LC 3915 OK")


if __name__ == "__main__":
    _test()
