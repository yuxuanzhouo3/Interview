"""
LC 517. Super Washing Machines — Hard
https://leetcode.com/problems/super-washing-machines/
Best: Prefix flow imbalance — O(n)
"""

from typing import List


class Solution:
    def findMinMoves(self, machines: List[int]) -> int:
        total = sum(machines)
        n = len(machines)
        if total % n:
            return -1
        avg = total // n
        ans = prefix = 0
        for m in machines:
            diff = m - avg
            prefix += diff
            ans = max(ans, abs(prefix), diff)
        return ans


def _test():
    s = Solution()
    assert s.findMinMoves([1, 0, 5]) == 3
    assert s.findMinMoves([0, 3, 0]) == 2
    assert s.findMinMoves([0, 2, 0]) == -1
    assert s.findMinMoves([4, 0, 0, 4]) == 2  # or check carefully
    print("LC 517 OK")


if __name__ == "__main__":
    _test()
