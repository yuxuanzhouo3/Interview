"""
LC 1998. GCD Sort of an Array — Hard
https://leetcode.com/problems/gcd-sort-of-an-array/
Best: SPF sieve + UnionFind on factors — O(n log A + A log log A)
"""

from typing import List


class Solution:
    def gcdSort(self, nums: List[int]) -> bool:
        mx = max(nums)
        spf = list(range(mx + 1))
        for i in range(2, int(mx**0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, mx + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for x in nums:
            v = x
            while v > 1:
                p = spf[v]
                union(x, p)
                while v % p == 0:
                    v //= p

        for a, b in zip(nums, sorted(nums)):
            if find(a) != find(b):
                return False
        return True


def _test():
    s = Solution()
    assert s.gcdSort([7, 21, 3]) is True
    assert s.gcdSort([5, 2, 6, 2]) is False
    assert s.gcdSort([10, 5, 9, 3, 15]) is True
    print("LC 1998 OK")


if __name__ == "__main__":
    _test()
