"""
LC 2076. Process Restricted Friend Requests — Hard
https://leetcode.com/problems/process-restricted-friend-requests/
Best: process in order + UF, reject if would merge a restriction — O(q·r·α)
"""

from typing import List


class Solution:
    def friendRequests(
        self, n: int, restrictions: List[List[int]], requests: List[List[int]]
    ) -> List[bool]:
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        ans = []
        for u, v in requests:
            pu, pv = find(u), find(v)
            ok = True
            if pu != pv:
                for x, y in restrictions:
                    px, py = find(x), find(y)
                    if (px == pu and py == pv) or (px == pv and py == pu):
                        ok = False
                        break
            ans.append(ok)
            if ok:
                union(u, v)
        return ans


def _test():
    s = Solution()
    assert s.friendRequests(3, [[0, 1]], [[0, 2], [2, 1]]) == [True, False]
    assert s.friendRequests(3, [[0, 1]], [[1, 2], [0, 2]]) == [True, False]
    assert s.friendRequests(5, [[0, 1], [1, 2], [2, 3]], [[0, 4], [1, 2], [3, 1], [3, 4]]) == [
        True,
        False,
        True,
        False,
    ]
    print("LC 2076 OK")


if __name__ == "__main__":
    _test()
