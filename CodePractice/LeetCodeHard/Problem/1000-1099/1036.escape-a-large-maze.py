"""
LC 1036. Escape a Large Maze — Hard
https://leetcode.com/problems/escape-a-large-maze/
Best: Bidirectional BFS limited by blocked area — O(B^2)
"""

from typing import List, Set, Tuple
from collections import deque


class Solution:
    def isEscapePossible(
        self, blocked: List[List[int]], source: List[int], target: List[int]
    ) -> bool:
        if not blocked:
            return True
        blocked_set = {tuple(p) for p in blocked}
        # max enclosed area by B blocks < B*(B-1)/2
        limit = len(blocked) * (len(blocked) - 1) // 2 + 1
        N = 10**6

        def can_reach(start: List[int], end: List[int]) -> bool:
            end_t = tuple(end)
            q = deque([tuple(start)])
            seen: Set[Tuple[int, int]] = {tuple(start)}
            while q and len(seen) <= limit:
                r, c = q.popleft()
                if (r, c) == end_t:
                    return True
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in blocked_set and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            return len(seen) > limit

        return can_reach(source, target) and can_reach(target, source)


def _test():
    s = Solution()
    assert s.isEscapePossible([[0, 1], [1, 0]], [0, 0], [0, 2]) is False
    assert s.isEscapePossible([], [0, 0], [999999, 999999]) is True
    assert s.isEscapePossible([[0, 3], [1, 1], [1, 2], [1, 3], [2, 1]], [0, 0], [0, 2]) is True
    print("LC 1036 OK")


if __name__ == "__main__":
    _test()
