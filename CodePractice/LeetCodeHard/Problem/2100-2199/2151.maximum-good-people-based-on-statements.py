"""
LC 2151. Maximum Good People Based on Statements — Hard
https://leetcode.com/problems/maximum-good-people-based-on-statements/
Best: enum good-bitmasks, validate — O(2^n * n^2)
"""

from typing import List


class Solution:
    def maximumGood(self, statements: List[List[int]]) -> int:
        n = len(statements)
        ans = 0
        for mask in range(1 << n):
            ok = True
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                for j in range(n):
                    st = statements[i][j]
                    if st == 2:
                        continue
                    good_j = 1 if mask & (1 << j) else 0
                    if st != good_j:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                ans = max(ans, bin(mask).count("1"))
        return ans


def _test():
    s = Solution()
    assert s.maximumGood([[2, 1, 2], [1, 2, 2], [2, 0, 2]]) == 2
    assert s.maximumGood([[2, 0], [0, 2]]) == 1
    print("LC 2151 OK")


if __name__ == "__main__":
    _test()
