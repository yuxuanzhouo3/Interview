"""
LC 1298. Maximum Candies You Can Get from Boxes — Hard
https://leetcode.com/problems/maximum-candies-you-can-get-from-boxes/
Best: BFS / queue process openable boxes — O(n + edges)
"""

from collections import deque
from typing import List


class Solution:
    def maxCandies(
        self,
        status: List[int],
        candies: List[int],
        keys: List[List[int]],
        containedBoxes: List[List[int]],
        initialBoxes: List[int],
    ) -> int:
        n = len(status)
        have = [False] * n
        opened = [False] * n
        q = deque()
        for b in initialBoxes:
            have[b] = True
            if status[b] == 1:
                q.append(b)
        ans = 0
        while q:
            b = q.popleft()
            if opened[b]:
                continue
            opened[b] = True
            ans += candies[b]
            for k in keys[b]:
                status[k] = 1
                if have[k] and not opened[k]:
                    q.append(k)
            for nb in containedBoxes[b]:
                have[nb] = True
                if status[nb] == 1 and not opened[nb]:
                    q.append(nb)
        return ans


def _test():
    s = Solution()
    assert s.maxCandies(
        [1, 0, 1, 0],
        [7, 5, 4, 100],
        [[], [], [1], []],
        [[1, 2], [3], [], []],
        [0],
    ) == 16
    assert s.maxCandies(
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [[1, 2, 3, 4, 5], [], [], [], [], []],
        [[1, 2, 3, 4, 5], [], [], [], [], []],
        [0],
    ) == 6
    assert s.maxCandies([1, 1, 1], [100, 1, 100], [[], [0, 2], []], [[], [], []], [1]) == 1
    print("LC 1298 OK")


if __name__ == "__main__":
    _test()
