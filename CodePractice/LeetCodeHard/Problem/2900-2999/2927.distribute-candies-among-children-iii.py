"""
LC 2927. Distribute Candies Among Children III — Hard
https://leetcode.com/problems/distribute-candies-among-children-iii/
Best: Stars and bars + inclusion-exclusion — O(1)
"""

class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def C2(x: int) -> int:
            return x * (x - 1) // 2 if x >= 2 else 0

        # x1+x2+x3=n, 0<=xi<=limit
        return C2(n + 2) - 3 * C2(n - limit + 1) + 3 * C2(n - 2 * limit) - C2(n - 3 * limit - 1)


def _test():
    s = Solution()
    assert s.distributeCandies(5, 2) == 3
    assert s.distributeCandies(3, 3) == 10
    print("LC 2927 OK")


if __name__ == "__main__":
    _test()

