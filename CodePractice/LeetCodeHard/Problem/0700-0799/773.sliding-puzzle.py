"""
LC 773. Sliding Puzzle — Hard
Best: BFS on board states — O(states)
"""

from typing import List
from collections import deque


class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        start = "".join(str(x) for row in board for x in row)
        target = "123450"
        # neighbors of index in 2x3
        neigh = {0: (1, 3), 1: (0, 2, 4), 2: (1, 5), 3: (0, 4), 4: (1, 3, 5), 5: (2, 4)}
        q = deque([(start, start.index("0"), 0)])
        seen = {start}
        while q:
            state, z, d = q.popleft()
            if state == target:
                return d
            for nz in neigh[z]:
                arr = list(state)
                arr[z], arr[nz] = arr[nz], arr[z]
                nxt = "".join(arr)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, nz, d + 1))
        return -1


def _test():
    s = Solution()
    assert s.slidingPuzzle([[1, 2, 3], [4, 0, 5]]) == 1
    assert s.slidingPuzzle([[1, 2, 3], [5, 4, 0]]) == -1
    print("LC 773 OK")


if __name__ == "__main__":
    _test()
