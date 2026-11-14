"""
LC 527. Word Abbreviation — Hard
https://leetcode.com/problems/word-abbreviation/
Best: Group by abbr + longest common prefix — O(n * L)
"""

from typing import List
from collections import defaultdict


class Solution:
    def wordsAbbreviation(self, words: List[str]) -> List[str]:
        def abbr(w: str, prefix_len: int) -> str:
            # keep first prefix_len chars + count + last
            if len(w) - prefix_len - 1 <= 1:
                return w
            return w[:prefix_len] + str(len(w) - prefix_len - 1) + w[-1]

        n = len(words)
        # initial abbreviation with prefix length 1
        pref = [1] * n
        ans = [abbr(w, 1) for w in words]

        while True:
            groups = defaultdict(list)
            for i, a in enumerate(ans):
                groups[a].append(i)
            conflict = False
            for indices in groups.values():
                if len(indices) <= 1:
                    continue
                conflict = True
                for i in indices:
                    pref[i] += 1
                    ans[i] = abbr(words[i], pref[i])
            if not conflict:
                break
        return ans


def _test():
    s = Solution()
    words = ["like", "god", "internal", "me", "internet", "interval", "intension", "face", "intrusion"]
    got = s.wordsAbbreviation(words)
    assert got == [
        "l2e",
        "god",
        "internal",
        "me",
        "i6t",
        "interval",
        "inte4n",
        "f2e",
        "intr4n",
    ]
    assert s.wordsAbbreviation(["aa", "aaa"]) == ["aa", "aaa"]
    print("LC 527 OK")


if __name__ == "__main__":
    _test()
