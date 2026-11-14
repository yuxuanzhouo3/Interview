"""
LC 2092. Find All People With Secret — Hard
https://leetcode.com/problems/find-all-people-with-secret/
Best: group by time + Union-Find — O(m log m + n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        union(0, firstPerson)
        meetings.sort(key=lambda x: x[2])
        i = 0
        m = len(meetings)
        while i < m:
            t = meetings[i][2]
            pool = []
            while i < m and meetings[i][2] == t:
                a, b, _ = meetings[i]
                union(a, b)
                pool.append(a)
                pool.append(b)
                i += 1
            # disconnect those who never got secret this round
            root0 = find(0)
            for p in pool:
                if find(p) != root0:
                    parent[p] = p
        return [i for i in range(n) if find(i) == find(0)]


def _test():
    s = Solution()
    assert sorted(s.findAllPeople(6, [[1, 2, 5], [2, 3, 8], [1, 5, 10]], 1)) == [0, 1, 2, 3, 5]
    assert sorted(s.findAllPeople(4, [[3, 1, 3], [1, 2, 2], [0, 3, 3]], 3)) == [0, 1, 3]
    assert sorted(s.findAllPeople(5, [[3, 4, 2], [1, 2, 1], [2, 3, 1]], 1)) == [0, 1, 2, 3, 4]
    print("LC 2092 OK")


if __name__ == "__main__":
    _test()
