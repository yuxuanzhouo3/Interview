"""
LC 2350. Shortest Impossible Sequence of Rolls — Hard
https://leetcode.com/problems/shortest-impossible-sequence-of-rolls/
Best: Greedy collect k distinct — O(n)
"""

from typing import List


class Solution:
    def shortestSequence(self, rolls: List[int], k: int) -> int:
        seen = set()
        groups = 0
        for x in rolls:
            if x not in seen:
                seen.add(x)
                if len(seen) == k:
                    groups += 1
                    seen.clear()
        return groups + 1


def _test():
    s = Solution()
    assert s.shortestSequence([4, 2, 1, 2, 3, 3, 2, 4, 1], 4) == 3
    assert s.shortestSequence([1, 1, 2, 2], 2) == 2
    assert s.shortestSequence([1, 1, 3, 2, 2, 2, 3, 3], 4) == 1
    print("LC 2350 OK")


if __name__ == "__main__":
    _test()
