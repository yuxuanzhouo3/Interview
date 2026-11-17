"""
LC 1255. Maximum Score Words Formed by Letters — Hard
https://leetcode.com/problems/maximum-score-words-formed-by-letters/
Best: Bitmask subset DFS/DP — O(2^n · Σ|w|)
"""

from typing import List


class Solution:
    def maxScoreWords(self, words: List[str], letters: List[str], score: List[int]) -> int:
        n = len(words)
        avail = [0] * 26
        for c in letters:
            avail[ord(c) - 97] += 1
        wcnt = []
        wscore = []
        for w in words:
            cnt = [0] * 26
            sc = 0
            for c in w:
                i = ord(c) - 97
                cnt[i] += 1
                sc += score[i]
            wcnt.append(cnt)
            wscore.append(sc)

        best = 0

        def dfs(i: int, cur: int, left: list) -> None:
            nonlocal best
            if i == n:
                best = max(best, cur)
                return
            dfs(i + 1, cur, left)
            ok = True
            for j in range(26):
                if wcnt[i][j] > left[j]:
                    ok = False
                    break
            if ok:
                for j in range(26):
                    left[j] -= wcnt[i][j]
                dfs(i + 1, cur + wscore[i], left)
                for j in range(26):
                    left[j] += wcnt[i][j]

        dfs(0, 0, avail)
        return best


def _test():
    s = Solution()
    assert s.maxScoreWords(
        ["dog", "cat", "dad", "good"],
        ["a", "a", "c", "d", "d", "d", "g", "o", "o"],
        [1, 0, 9, 5, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ) == 23
    assert s.maxScoreWords(
        ["xxxz", "ax", "bx", "cx"],
        ["z", "a", "b", "c", "x", "x", "x"],
        [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 10],
    ) == 27
    assert s.maxScoreWords(
        ["leetcode"],
        list("letcod"),
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ) == 0
    print("LC 1255 OK")


if __name__ == "__main__":
    _test()
