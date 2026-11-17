"""
LC 2157. Groups of Strings — Hard
https://leetcode.com/problems/groups-of-strings/
Best: Bitmask UF connect edit-distance≤2 — O(n*26^2)
"""

from typing import List
from collections import defaultdict


class Solution:
    def groupStrings(self, words: List[str]) -> List[int]:
        parent = {}
        size = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        cnt = defaultdict(int)
        for w in words:
            m = 0
            for c in w:
                m |= 1 << (ord(c) - 97)
            cnt[m] += 1

        for m, c in cnt.items():
            parent[m] = m
            size[m] = c

        present = set(cnt)
        for m in list(present):
            for i in range(26):
                if m & (1 << i):
                    nm = m ^ (1 << i)
                    if nm in present:
                        union(m, nm)
                    for j in range(26):
                        if (m & (1 << j)) == 0:
                            rm = nm | (1 << j)
                            if rm in present:
                                union(m, rm)
                else:
                    nm = m | (1 << i)
                    if nm in present:
                        union(m, nm)

        seen = set()
        groups = 0
        maxsz = 0
        for m in present:
            r = find(m)
            if r not in seen:
                seen.add(r)
                groups += 1
                maxsz = max(maxsz, size[r])
        return [groups, maxsz]


def _test():
    s = Solution()
    assert s.groupStrings(["a", "b", "ab", "cde"]) == [2, 3]
    assert s.groupStrings(["a", "ab", "abc"]) == [1, 3]
    assert s.groupStrings(["a", "b", "ab", "ba"]) == [1, 4]
    print("LC 2157 OK")


if __name__ == "__main__":
    _test()
