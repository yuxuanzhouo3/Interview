"""
LC 1916. Count Ways to Build Rooms in an Ant Colony — Hard
https://leetcode.com/problems/count-ways-to-build-rooms-in-an-ant-colony/
Best: Tree DP + multinomial — O(n)
"""

from collections import defaultdict
from math import comb
from typing import List

class Solution:
    def waysToBuildRooms(self, prevRoom: List[int]) -> int:
        MOD = 10**9 + 7
        g = defaultdict(list)
        for i in range(1, len(prevRoom)):
            g[prevRoom[i]].append(i)
        ans = 1

        def dfs(u: int) -> int:
            nonlocal ans
            total = 0
            for v in g[u]:
                sz = dfs(v)
                if total:
                    ans = ans * comb(total + sz, sz) % MOD
                total += sz
            return total + 1

        dfs(0)
        return ans % MOD

def _test():

    s = Solution()
    assert s.waysToBuildRooms([-1, 0, 1]) == 1
    assert s.waysToBuildRooms([-1, 0, 0, 1, 2]) == 6
    print("LC 1916 OK")

if __name__ == "__main__":
    _test()
