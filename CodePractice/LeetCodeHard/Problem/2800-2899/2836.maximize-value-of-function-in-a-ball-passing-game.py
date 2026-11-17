"""
LC 2836. Maximize Value of Function in a Ball Passing Game — Hard
https://leetcode.com/problems/maximize-value-of-function-in-a-ball-passing-game/
Best: Binary lifting on functional graph — O(n log k)
"""

from typing import List


class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        LOG = k.bit_length()
        up = [[0] * n for _ in range(LOG)]
        sm = [[0] * n for _ in range(LOG)]
        for i in range(n):
            up[0][i] = receiver[i]
            sm[0][i] = receiver[i]
        for j in range(1, LOG):
            for i in range(n):
                mid = up[j - 1][i]
                up[j][i] = up[j - 1][mid]
                sm[j][i] = sm[j - 1][i] + sm[j - 1][mid]
        ans = 0
        for i in range(n):
            cur, total, steps = i, i, k
            for j in range(LOG):
                if steps & (1 << j):
                    total += sm[j][cur]
                    cur = up[j][cur]
            ans = max(ans, total)
        return ans


def _test():
    s = Solution()
    assert s.getMaxFunctionValue([2, 0, 1], 4) == 6
    assert s.getMaxFunctionValue([1, 1, 1, 2, 3], 3) == 10
    print("LC 2836 OK")


if __name__ == "__main__":
    _test()
