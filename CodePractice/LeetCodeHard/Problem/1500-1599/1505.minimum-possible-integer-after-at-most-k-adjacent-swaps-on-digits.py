"""
LC 1505. Minimum Possible Integer After at Most K Adjacent Swaps On Digits — Hard
https://leetcode.com/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits/
Best: Greedy + Fenwick for cost — O(n log n)
"""

from collections import deque


class BIT:
    def __init__(self, n):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i, v):
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def sum(self, i):
        s = 0
        while i:
            s += self.t[i]
            i -= i & -i
        return s


class Solution:
    def minInteger(self, num: str, k: int) -> str:
        n = len(num)
        pos = [deque() for _ in range(10)]
        for i, ch in enumerate(num):
            pos[int(ch)].append(i)
        bit = BIT(n)
        for i in range(n):
            bit.add(i + 1, 1)
        ans = []
        for _ in range(n):
            for d in range(10):
                if not pos[d]:
                    continue
                idx = pos[d][0]
                # remaining indices strictly before idx
                cost = bit.sum(idx)
                if cost <= k:
                    k -= cost
                    ans.append(str(d))
                    pos[d].popleft()
                    bit.add(idx + 1, -1)
                    break
        return "".join(ans)


def _test():
    s = Solution()
    assert s.minInteger("4321", 4) == "1342"
    assert s.minInteger("100", 1) == "010"
    assert s.minInteger("36789", 1000) == "36789"
    print("LC 1505 OK")


if __name__ == "__main__":
    _test()
