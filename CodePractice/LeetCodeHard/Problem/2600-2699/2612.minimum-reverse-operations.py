"""
LC 2612. Minimum Reverse Operations — Hard
https://leetcode.com/problems/minimum-reverse-operations/
Best: BFS + SortedSet parity jumps — O(n log n)
"""

from collections import deque
from typing import List
from sortedcontainers import SortedSet

class Solution:
    def minReverseOperations(self, n: int, p: int, banned: List[int], k: int) -> List[int]:
        ans = [-1] * n
        ans[p] = 0
        ts = [SortedSet() for _ in range(2)]
        for i in range(n):
            ts[i % 2].add(i)
        ts[p % 2].remove(p)
        for i in banned:
            ts[i % 2].remove(i)
        ts[0].add(n)
        ts[1].add(n)
        q = deque([p])
        while q:
            i = q.popleft()
            mi = max(i - k + 1, k - i - 1)
            mx = min(i + k - 1, n * 2 - k - i - 1)
            s = ts[mi % 2]
            j = s.bisect_left(mi)
            while s[j] <= mx:
                q.append(s[j])
                ans[s[j]] = ans[i] + 1
                s.remove(s[j])
                j = s.bisect_left(mi)
        return ans

def _test():

    s = Solution()
    assert s.minReverseOperations(4, 0, [1, 2], 4) == [0, -1, -1, 1]
    assert s.minReverseOperations(5, 0, [2, 4], 3) == [0, -1, -1, -1, -1]
    assert s.minReverseOperations(4, 2, [0, 1, 3], 1) == [-1, -1, 0, -1]
    print("LC 2612 OK")

if __name__ == "__main__":
    _test()
