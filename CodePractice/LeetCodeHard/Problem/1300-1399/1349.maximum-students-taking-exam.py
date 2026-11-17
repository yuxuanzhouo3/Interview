"""
LC 1349. Maximum Students Taking Exam — Hard
https://leetcode.com/problems/maximum-students-taking-exam/
Best: Bitmask DP on seats — O(m · 2^{2n})
"""

from typing import List


class Solution:
    def maxStudents(self, seats: List[List[str]]) -> int:
        m, n = len(seats), len(seats[0])
        valid = []
        for row in seats:
            mask = 0
            for j, c in enumerate(row):
                if c == ".":
                    mask |= 1 << j
            valid.append(mask)

        def ok(mask: int, rowmask: int) -> bool:
            if mask & ~rowmask:
                return False
            if mask & (mask << 1):
                return False
            return True

        # dp[mask] for previous row
        dp = {-1: 0}  # use dict: prev_mask -> max students
        dp = {0: 0}
        for r in range(m):
            ndp = {}
            for pmask, val in dp.items():
                for mask in range(1 << n):
                    if not ok(mask, valid[r]):
                        continue
                    if (mask & (pmask << 1)) or (mask & (pmask >> 1)):
                        continue
                    ndp[mask] = max(ndp.get(mask, 0), val + bin(mask).count("1"))
            dp = ndp
        return max(dp.values()) if dp else 0


def _test():
    s = Solution()
    assert s.maxStudents(
        [["#", ".", "#", "#", ".", "#"],
         [".", "#", "#", "#", "#", "."],
         ["#", ".", "#", "#", ".", "#"]]
    ) == 4
    assert s.maxStudents([[".", "#"]]) == 1
    assert s.maxStudents(
        [["#", ".", ".", ".", "#"],
         [".", "#", ".", "#", "."],
         [".", ".", "#", ".", "."],
         [".", "#", ".", "#", "."],
         ["#", ".", ".", ".", "#"]]
    ) == 10
    print("LC 1349 OK")


if __name__ == "__main__":
    _test()
