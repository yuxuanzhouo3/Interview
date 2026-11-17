"""
LC 964. Least Operators to Express Number — Hard
https://leetcode.com/problems/least-operators-to-express-number/
Best: Recurse on nearest x-powers with shrink guard — O(log_x target)
"""

from functools import lru_cache


class Solution:
    def leastOpsExpressTarget(self, x: int, target: int) -> int:
        @lru_cache(None)
        def dfs(t: int) -> int:
            if t == 0:
                return 0
            if t < x:
                return min(2 * t - 1, 2 * (x - t))
            p, y = 0, x
            while y * x <= t:
                y *= x
                p += 1
            # y = x^{p+1}; forming y costs p operators (*'s)
            if t == y:
                return p
            ans = dfs(t - y) + p + 1  # + between y and remainder
            nxt = y * x
            if nxt - t < t:
                ans = min(ans, dfs(nxt - t) + p + 2)
            return ans

        return dfs(target)


def _test():
    s = Solution()
    assert s.leastOpsExpressTarget(3, 19) == 5
    assert s.leastOpsExpressTarget(5, 501) == 8
    assert s.leastOpsExpressTarget(100, 100000000) == 3
    print("LC 964 OK")


if __name__ == "__main__":
    _test()
