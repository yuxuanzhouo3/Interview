"""
LC 1788. Maximize the Beauty of the Garden — Hard
Best: First occurrence + positive prefix — O(n)
"""

from typing import List
from math import inf


class Solution:
    def maximumBeauty(self, flowers: List[int]) -> int:
        s = [0] * (len(flowers) + 1)
        d = {}
        ans = -inf
        for i, v in enumerate(flowers):
            if v in d:
                ans = max(ans, s[i] - s[d[v] + 1] + v * 2)
            else:
                d[v] = i
            s[i + 1] = s[i] + max(v, 0)
        return int(ans)


def _test():
    sol = Solution()
    assert sol.maximumBeauty([1, 2, 3, 1, 2]) == 8
    assert sol.maximumBeauty([100, 1, 1, -3, 1]) == 3
    assert sol.maximumBeauty([-1, -2, 0, -1]) == -2
    print("LC 1788 OK")


if __name__ == "__main__":
    _test()
