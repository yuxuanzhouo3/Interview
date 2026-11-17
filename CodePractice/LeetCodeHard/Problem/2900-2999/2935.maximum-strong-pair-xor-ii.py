"""
LC 2935. Maximum Strong Pair XOR II — Hard
https://leetcode.com/problems/maximum-strong-pair-xor-ii/
Best: Sort + sliding window 01-Trie — O(n * 20)
"""

from typing import List


class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        nums.sort()

        class Node:
            __slots__ = ("c", "cnt")

            def __init__(self):
                self.c = [None, None]
                self.cnt = 0

        root = Node()

        def tog(x: int, delta: int) -> None:
            p = root
            for b in range(20, -1, -1):
                bit = (x >> b) & 1
                if p.c[bit] is None:
                    p.c[bit] = Node()
                p = p.c[bit]
                p.cnt += delta

        def query(x: int) -> int:
            p = root
            res = 0
            for b in range(20, -1, -1):
                bit = (x >> b) & 1
                want = 1 - bit
                if p.c[want] is not None and p.c[want].cnt > 0:
                    res |= 1 << b
                    p = p.c[want]
                elif p.c[bit] is not None and p.c[bit].cnt > 0:
                    p = p.c[bit]
                else:
                    return 0
            return res

        ans = 0
        j = 0
        for y in nums:
            tog(y, 1)
            while nums[j] * 2 < y:
                tog(nums[j], -1)
                j += 1
            ans = max(ans, query(y))
        return ans


def _test():
    s = Solution()
    assert s.maximumStrongPairXor([1, 2, 3, 4, 5]) == 7
    assert s.maximumStrongPairXor([10, 100]) == 0
    assert s.maximumStrongPairXor([500, 520, 510]) == 1020
    print("LC 2935 OK")


if __name__ == "__main__":
    _test()
