"""
LC 1178. Number of Valid Words for Each Puzzle — Hard
https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/
Best: Bitmask + enumerate subsets — O(W + P·2^|unique|)
"""

from collections import Counter
from typing import List


class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        cnt = Counter()
        for w in words:
            mask = 0
            for c in w:
                mask |= 1 << (ord(c) - 97)
            if bin(mask).count("1") <= 7:
                cnt[mask] += 1

        ans = []
        for p in puzzles:
            first = 1 << (ord(p[0]) - 97)
            mask = 0
            for c in p:
                mask |= 1 << (ord(c) - 97)
            total = 0
            sub = mask
            while sub:
                if sub & first:
                    total += cnt[sub]
                sub = (sub - 1) & mask
            ans.append(total)
        return ans


def _test():
    s = Solution()
    assert s.findNumOfValidWords(
        ["aaaa", "asas", "able", "ability", "actt", "actor", "access"],
        ["aboveyz", "abrodyz", "abslute", "absoryz", "actresz", "gaswxyz"],
    ) == [1, 1, 3, 2, 4, 0]
    assert s.findNumOfValidWords(
        ["apple", "pleas", "please"],
        ["aelwxyz", "aelpxyz", "aelpsxy", "saelpxy", "xaelpsy"],
    ) == [0, 1, 3, 2, 0]
    print("LC 1178 OK")


if __name__ == "__main__":
    _test()
