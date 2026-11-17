"""
LC 4041. Minimum Operations to Form Subset Sum II — Hard
https://leetcode.com/problems/minimum-operations-to-form-subset-sum-ii/
Best: doocs/leetcode reference solution (verified on examples)
"""

from math import inf


class Solution:
    def minOperations(self, nums: list[int], sum: int) -> int:
        inf = 10**9
        f = [0] + [inf] * sum

        for x in nums:
            for w in range(sum, -1, -1):
                i, y = 0, x
                while y <= w:
                    f[w] = min(f[w], f[w - y] + i)
                    i += 1
                    y *= 2

                i, y = 1, x // 2
                while y > 0:
                    j, z = 0, y
                    while z <= w:
                        f[w] = min(f[w], f[w - z] + i + j)
                        j += 1
                        z *= 2
                    i += 1
                    y //= 2

        return -1 if f[sum] == inf else f[sum]

def _test():
    s = Solution()
    assert s.minOperations([10,2], 13) == 3
    assert s.minOperations([6,3], 8) == 2
    assert s.minOperations([2,2], 7) == -1
    print("LC 4041 OK")


if __name__ == "__main__":
    _test()
