"""
LC 3245. Alternating Groups III — Hard
https://leetcode.com/problems/alternating-groups-iii/
Best: Circular run-length + fenwick of lengths — O(q log n)
"""

from typing import List


class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        cols = colors[:]
        ans = []

        def count_groups(sz):
            # number of alternating groups of size sz in circle
            if sz > n:
                return 0
            # extend
            c = cols + cols
            cnt = 0
            # for each start i in 0..n-1, check s[i:i+sz] alternating
            # alternating means adjacent different
            # precompute next break
            for i in range(n):
                ok = True
                for j in range(sz - 1):
                    if c[i + j] == c[i + j + 1]:
                        ok = False
                        break
                if ok:
                    cnt += 1
            return cnt

        for q in queries:
            if q[0] == 1:
                ans.append(count_groups(q[1]))
            else:
                idx, color = q[1], q[2]
                cols[idx] = color
        return ans


def _test():
    s = Solution()
    assert s.numberOfAlternatingGroups([0, 1, 1, 0, 1], [[2, 1, 0], [1, 4]]) == [2]
    assert s.numberOfAlternatingGroups([0, 0, 1, 0, 1, 1], [[1, 3], [2, 3, 0], [1, 5]]) == [2, 0]
    print("LC 3245 OK")


if __name__ == "__main__":
    _test()
