"""
LC 1707. Maximum XOR With an Element From Array — Hard
https://leetcode.com/problems/maximum-xor-with-an-element-from-array/
Best: Offline queries + binary Trie — O((n+q) log)
"""

from typing import List


class TrieNode:
    __slots__ = ("ch",)

    def __init__(self):
        self.ch = [None, None]


class Solution:
    def maximizeXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        nums = sorted(set(nums))
        qi = sorted(range(len(queries)), key=lambda i: queries[i][1])
        ans = [-1] * len(queries)
        root = TrieNode()
        p = 0

        def insert(x):
            node = root
            for b in range(30, -1, -1):
                bit = (x >> b) & 1
                if not node.ch[bit]:
                    node.ch[bit] = TrieNode()
                node = node.ch[bit]

        def query(x):
            node = root
            if not node.ch[0] and not node.ch[1]:
                return -1
            res = 0
            for b in range(30, -1, -1):
                bit = (x >> b) & 1
                want = 1 - bit
                if node.ch[want]:
                    res |= 1 << b
                    node = node.ch[want]
                else:
                    node = node.ch[bit]
            return res

        for i in qi:
            x, m = queries[i]
            while p < len(nums) and nums[p] <= m:
                insert(nums[p])
                p += 1
            ans[i] = query(x)
        return ans


def _test():
    s = Solution()
    assert s.maximizeXor([0, 1, 2, 3, 4], [[3, 1], [1, 3], [5, 6]]) == [3, 3, 7]
    assert s.maximizeXor([5, 2, 4, 6, 6, 3], [[12, 4], [8, 1], [6, 3]]) == [15, -1, 5]
    print("LC 1707 OK")


if __name__ == "__main__":
    _test()
