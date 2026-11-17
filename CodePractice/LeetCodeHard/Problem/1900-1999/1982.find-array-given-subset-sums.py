"""
LC 1982. Find Array Given Subset Sums — Hard
https://leetcode.com/problems/find-array-given-subset-sums/
Best: Recover from subset sums via SortedList — O(2^n · n)
"""

from typing import List
from sortedcontainers import SortedList

class Solution:
    def recoverArray(self, n: int, sums: List[int]) -> List[int]:
        m = -min(sums)
        sl = SortedList(x + m for x in sums)
        sl.remove(0)
        ans = [sl[0]]
        for i in range(1, n):
            for j in range(1 << i):
                if j >> (i - 1) & 1:
                    s = sum(ans[k] for k in range(i) if j >> k & 1)
                    sl.remove(s)
            ans.append(sl[0])
        for i in range(1 << n):
            s = sum(ans[j] for j in range(n) if i >> j & 1)
            if s == m:
                for j in range(n):
                    if i >> j & 1:
                        ans[j] *= -1
                break
        return ans

def _test():

    def check(n, sums, ans):
        from itertools import combinations
        assert len(ans) == n
        got = []
        for mask in range(1 << n):
            got.append(sum(ans[i] for i in range(n) if mask >> i & 1))
        assert sorted(got) == sorted(sums)

    s = Solution()
    for n, sums in [
        (3, [-3, -2, -1, 0, 0, 1, 2, 3]),
        (2, [0, 0, 0, 0]),
        (4, [0, 0, 5, 5, 4, -1, 4, 9, 9, -1, 4, 3, 4, 8, 3, 8]),
    ]:
        ans = s.recoverArray(n, sums)
        check(n, sums, ans)
    print("LC 1982 OK")

if __name__ == "__main__":
    _test()
