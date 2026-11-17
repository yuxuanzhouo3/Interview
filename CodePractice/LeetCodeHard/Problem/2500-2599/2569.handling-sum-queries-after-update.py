"""
LC 2569. Handling Sum Queries After Update — Hard
https://leetcode.com/problems/handling-sum-queries-after-update/
Best: Segment tree lazy flip — O((n+q) log n)
"""

from typing import List


class Solution:
    def handleQuery(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        tree = [0] * (4 * n)
        lazy = [0] * (4 * n)

        def build(i: int, l: int, r: int) -> None:
            if l == r:
                tree[i] = nums1[l]
                return
            m = (l + r) // 2
            build(2 * i, l, m)
            build(2 * i + 1, m + 1, r)
            tree[i] = tree[2 * i] + tree[2 * i + 1]

        def apply(i: int, l: int, r: int) -> None:
            tree[i] = (r - l + 1) - tree[i]
            lazy[i] ^= 1

        def push(i: int, l: int, r: int) -> None:
            if lazy[i] and l != r:
                m = (l + r) // 2
                apply(2 * i, l, m)
                apply(2 * i + 1, m + 1, r)
                lazy[i] = 0

        def upd(i: int, l: int, r: int, ql: int, qr: int) -> None:
            if qr < l or r < ql:
                return
            if ql <= l and r <= qr:
                apply(i, l, r)
                return
            push(i, l, r)
            m = (l + r) // 2
            upd(2 * i, l, m, ql, qr)
            upd(2 * i + 1, m + 1, r, ql, qr)
            tree[i] = tree[2 * i] + tree[2 * i + 1]

        build(1, 0, n - 1)
        s2 = sum(nums2)
        ans: List[int] = []
        for t, a, b in queries:
            if t == 1:
                upd(1, 0, n - 1, a, b)
            elif t == 2:
                s2 += tree[1] * a
            else:
                ans.append(s2)
        return ans


def _test():
    s = Solution()
    assert s.handleQuery([1, 0, 1], [0, 0, 0], [[1, 1, 1], [2, 1, 0], [3, 0, 0]]) == [3]
    assert s.handleQuery([1, 0, 1], [100, 100, 100], [[1, 1, 1], [2, 1, 0], [3, 0, 0]]) == [303]
    print("LC 2569 OK")


if __name__ == "__main__":
    _test()
