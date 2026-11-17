"""
LC 3307. Find the K-th Character in String Game II — Hard
https://leetcode.com/problems/find-the-k-th-character-in-string-game-ii/
Best: Reverse simulate on bit length — O(log k)
"""

from typing import List


class Solution:
    def kthCharacter(self, k: int, operations: List[int]) -> str:
        # start "a"; op0: append copy; op1: append next-letter copy
        k -= 1  # 0-index
        shift = 0
        # after len(operations) ops, length = 2^{len}
        for i in range(len(operations) - 1, -1, -1):
            half = 1 << i
            if k >= half:
                k -= half
                if operations[i] == 1:
                    shift += 1
        return chr(ord("a") + shift % 26)


def _test():
    s = Solution()
    assert s.kthCharacter(5, [0, 0, 0]) == "a"
    assert s.kthCharacter(10, [0, 1, 0, 1]) == "b"
    print("LC 3307 OK")


if __name__ == "__main__":
    _test()
