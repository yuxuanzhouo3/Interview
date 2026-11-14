"""
LC 1345. Jump Game IV — Hard
https://leetcode.com/problems/jump-game-iv/
Best: BFS + value→indices map — O(n)
"""

from typing import List
from collections import defaultdict, deque


class Solution:
    def minJumps(self, arr: List[int]) -> int:
        n = len(arr)
        if n == 1:
            return 0
        pos = defaultdict(list)
        for i, v in enumerate(arr):
            pos[v].append(i)
        q = deque([0])
        seen = {0}
        steps = 0
        while q:
            for _ in range(len(q)):
                i = q.popleft()
                if i == n - 1:
                    return steps
                for j in (i - 1, i + 1):
                    if 0 <= j < n and j not in seen:
                        seen.add(j)
                        q.append(j)
                for j in pos[arr[i]]:
                    if j not in seen:
                        seen.add(j)
                        q.append(j)
                pos[arr[i]].clear()
            steps += 1
        return -1


def _test():
    s = Solution()
    assert s.minJumps([100, -23, -23, 404, 100, 23, 23, 23, 3, 404]) == 3
    assert s.minJumps([7]) == 0
    assert s.minJumps([7, 6, 9, 6, 9, 6, 9, 7]) == 1
    print("LC 1345 OK")


if __name__ == "__main__":
    _test()
