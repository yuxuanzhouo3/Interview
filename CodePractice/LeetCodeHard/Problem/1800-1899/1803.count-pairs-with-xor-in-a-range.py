"""
LC 1803. Count Pairs With XOR in a Range — Hard
https://leetcode.com/problems/count-pairs-with-xor-in-a-range/
Best: 01-Trie count XOR < limit — O(n log A)
"""


from typing import List


class Trie:
    __slots__ = ("ch", "cnt")

    def __init__(self):
        self.ch = [None, None]
        self.cnt = 0

    def insert(self, x: int) -> None:
        node = self
        for i in range(15, -1, -1):
            b = (x >> i) & 1
            if node.ch[b] is None:
                node.ch[b] = Trie()
            node = node.ch[b]
            node.cnt += 1

    def count_less(self, x: int, limit: int) -> int:
        node = self
        ans = 0
        for i in range(15, -1, -1):
            if node is None:
                return ans
            xb = (x >> i) & 1
            if (limit >> i) & 1:
                if node.ch[xb]:
                    ans += node.ch[xb].cnt
                node = node.ch[xb ^ 1]
            else:
                node = node.ch[xb]
        return ans


class Solution:
    def countPairs(self, nums: List[int], low: int, high: int) -> int:
        trie = Trie()
        ans = 0
        for x in nums:
            ans += trie.count_less(x, high + 1) - trie.count_less(x, low)
            trie.insert(x)
        return ans


def _test():
    s = Solution()
    assert s.countPairs([1, 4, 2, 7], 2, 6) == 6
    assert s.countPairs([9, 8, 4, 2, 1], 5, 14) == 8
    print("LC 1803 OK")


if __name__ == "__main__":
    _test()

