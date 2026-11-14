"""
LC 403. Frog Jump — Hard
Best: DP set of reachable jump sizes per stone — O(n^2)
"""

from typing import List


class Solution:
    def canCross(self, stones: List[int]) -> bool:
        if stones[1] != 1:
            return False
        pos = {s: set() for s in stones}
        pos[0].add(0)
        stone_set = set(stones)
        for s in stones:
            for k in pos[s]:
                for dk in (k - 1, k, k + 1):
                    if dk > 0 and s + dk in stone_set:
                        pos[s + dk].add(dk)
        return bool(pos[stones[-1]])


def _test():
    s = Solution()
    assert s.canCross([0, 1, 3, 5, 6, 8, 12, 17]) is True
    assert s.canCross([0, 1, 2, 3, 4, 8, 9, 11]) is False
    print("LC 403 OK")


if __name__ == "__main__":
    _test()
