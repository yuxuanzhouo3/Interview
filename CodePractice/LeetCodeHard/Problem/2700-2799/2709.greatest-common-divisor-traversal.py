"""
LC 2709. Greatest Common Divisor Traversal — Hard
https://leetcode.com/problems/greatest-common-divisor-traversal/
Best: Union-Find on prime factors — O(n√A α)
"""

from typing import List


class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        prime_to_idx = {}

        def factors(x):
            res = []
            d = 2
            while d * d <= x:
                if x % d == 0:
                    res.append(d)
                    while x % d == 0:
                        x //= d
                d += 1
            if x > 1:
                res.append(x)
            return res

        for i, x in enumerate(nums):
            if x == 1:
                continue
            for p in factors(x):
                if p in prime_to_idx:
                    union(i, prime_to_idx[p])
                else:
                    prime_to_idx[p] = i

        if any(x == 1 for x in nums):
            return False
        root = find(0)
        return all(find(i) == root for i in range(n))


def _test():
    s = Solution()
    assert s.canTraverseAllPairs([2, 3, 6]) is True
    assert s.canTraverseAllPairs([3, 9, 5]) is False
    assert s.canTraverseAllPairs([4, 3, 12, 8]) is True
    assert s.canTraverseAllPairs([1]) is True
    print("LC 2709 OK")


if __name__ == "__main__":
    _test()
