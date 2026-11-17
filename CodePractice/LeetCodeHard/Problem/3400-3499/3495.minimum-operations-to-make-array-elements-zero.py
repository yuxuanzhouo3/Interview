"""
LC 3495. Minimum Operations to Make Array Elements Zero — Hard
https://leetcode.com/problems/minimum-operations-to-make-array-elements-zero/
Best: Digit / base-4 depth prefix sums — O(q log R)
"""

from typing import List


class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # op: replace x with floor(x/4). ops for one number = steps to 0
        # steps(x) = floor(log4(x))+1
        def steps_sum(n: int) -> int:
            # sum of steps(x) for x=1..n
            if n <= 0:
                return 0
            ans = 0
            start = 1
            level = 1
            while start <= n:
                end = min(n, start * 4 - 1)
                ans += (end - start + 1) * level
                start *= 4
                level += 1
            return ans

        def range_sum(l, r):
            return steps_sum(r) - steps_sum(l - 1)

        total = 0
        for l, r in queries:
            s = range_sum(l, r)
            # each op can reduce two numbers (pick two indices), so ceil(s/2)
            total += (s + 1) // 2
        return total


def _test():
    s = Solution()
    assert s.minOperations([[1, 2], [2, 4]]) == 3
    assert s.minOperations([[2, 6]]) == 4
    print("LC 3495 OK")


if __name__ == "__main__":
    _test()
