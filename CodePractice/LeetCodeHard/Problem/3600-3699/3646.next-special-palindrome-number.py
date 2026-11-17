"""
LC 3646. Next Special Palindrome Number — Hard
https://leetcode.com/problems/next-special-palindrome-number/
Best: Enumerate special palindromes + bisect — O(P log P)
"""

from __future__ import annotations

from typing import List, Optional
import bisect
from itertools import combinations


class Solution:
    _cands: Optional[List[int]] = None

    @classmethod
    def _build(cls) -> List[int]:
        if cls._cands is not None:
            return cls._cands
        cands: List[int] = []

        def unique_perms(items: List[int]):
            items = sorted(items)
            n = len(items)
            if n == 0:
                yield ()
                return
            used = [False] * n

            def dfs(path: List[int]):
                if len(path) == n:
                    yield tuple(path)
                    return
                for i in range(n):
                    if used[i]:
                        continue
                    if i and items[i] == items[i - 1] and not used[i - 1]:
                        continue
                    used[i] = True
                    path.append(items[i])
                    yield from dfs(path)
                    path.pop()
                    used[i] = False

            yield from dfs([])

        digits = list(range(1, 10))
        for r in range(1, 10):
            for subset in combinations(digits, r):
                if sum(subset) > 18:  # covers n <= 1e15
                    continue
                odds = [d for d in subset if d % 2 == 1]
                if len(odds) > 1:
                    continue
                half: List[int] = []
                mid = odds[0] if odds else None
                for d in subset:
                    half.extend([d] * (d // 2))
                for perm in unique_perms(half):
                    left = "".join(map(str, perm))
                    s = left + ("" if mid is None else str(mid)) + left[::-1]
                    cands.append(int(s))
        cls._cands = sorted(set(cands))
        return cls._cands

    def specialPalindrome(self, n: int) -> int:
        cands = self._build()
        return cands[bisect.bisect_right(cands, n)]


def _test():
    s = Solution()
    assert s.specialPalindrome(2) == 22
    assert s.specialPalindrome(33) == 212
    print("LC 3646 OK")


if __name__ == "__main__":
    _test()
