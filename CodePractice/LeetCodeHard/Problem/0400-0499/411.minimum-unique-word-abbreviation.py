"""
LC 411. Minimum Unique Word Abbreviation — Hard
https://leetcode.com/problems/minimum-unique-word-abbreviation/
Best: 位掩码枚举保留位 + 最短缩写 — O(2^n · m · n)
"""

from typing import List


class Solution:
    def minAbbreviation(self, target: str, dictionary: List[str]) -> str:
        n = len(target)
        diffs = []
        for w in dictionary:
            if len(w) != n:
                continue
            mask = 0
            for i in range(n):
                if w[i] != target[i]:
                    mask |= 1 << i
            if mask == 0:
                # 完全相同，无法缩写区分；按题意字典不含 target
                continue
            diffs.append(mask)
        if not diffs:
            return str(n) if n else ""

        best = target
        best_len = n + 1

        def abbr_len(keep: int) -> int:
            length = 0
            i = 0
            while i < n:
                if keep & (1 << i):
                    length += 1
                    i += 1
                else:
                    length += 1
                    while i < n and not (keep & (1 << i)):
                        i += 1
            return length

        def to_abbr(keep: int) -> str:
            parts = []
            i = 0
            while i < n:
                if keep & (1 << i):
                    parts.append(target[i])
                    i += 1
                else:
                    j = i
                    while j < n and not (keep & (1 << j)):
                        j += 1
                    parts.append(str(j - i))
                    i = j
            return "".join(parts)

        for keep in range(1 << n):
            if all(keep & d for d in diffs):
                ln = abbr_len(keep)
                if ln < best_len:
                    best_len = ln
                    best = to_abbr(keep)
        return best


def _test():
    s = Solution()
    assert s.minAbbreviation("apple", ["blade"]) == "a4"
    assert s.minAbbreviation("apple", []) == "5"
    assert s.minAbbreviation("apple", ["plain", "amber", "blade"]) in {
        "1p3",
        "2p2",
        "3le",
        "3l1",
        "a3e",
        "ap3",
    }
    assert s.minAbbreviation("apple", ["april"]) == "4e"
    print("LC 411 OK")


if __name__ == "__main__":
    _test()
