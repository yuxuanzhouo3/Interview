"""
LC 2407. Longest Increasing Subsequence II — Hard
https://leetcode.com/problems/longest-increasing-subsequence-ii/
Best: Segment tree max DP LIS with diff≤k — O(n log M)
"""


from typing import List


class SegTree:
    def __init__(self, n):
        self.n = n
        self.t = [0] * (4 * n)

    def update(self, idx, val, node=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        if l == r:
            self.t[node] = max(self.t[node], val)
            return
        m = (l + r) // 2
        if idx <= m:
            self.update(idx, val, node * 2, l, m)
        else:
            self.update(idx, val, node * 2 + 1, m + 1, r)
        self.t[node] = max(self.t[node * 2], self.t[node * 2 + 1])

    def query(self, ql, qr, node=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        if ql > r or qr < l or ql > qr:
            return 0
        if ql <= l and r <= qr:
            return self.t[node]
        m = (l + r) // 2
        return max(self.query(ql, qr, node * 2, l, m), self.query(ql, qr, node * 2 + 1, m + 1, r))


class Solution:
    def lengthOfLIS(self, nums: List[int], k: int) -> int:
        M = max(nums)
        st = SegTree(M + 2)
        ans = 0
        for x in nums:
            best = st.query(max(1, x - k), x - 1) + 1
            st.update(x, best)
            ans = max(ans, best)
        return ans


def _test():
    s = Solution()
    assert s.lengthOfLIS([4, 2, 1, 4, 3, 4, 5, 8, 15], 3) == 5
    assert s.lengthOfLIS([7, 4, 5, 1, 8, 12, 4, 7], 5) == 4
    assert s.lengthOfLIS([1, 5], 1) == 1
    print("LC 2407 OK")


if __name__ == "__main__":
    _test()

