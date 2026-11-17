"""
LC 2301. Match Substring After Replacement — Hard
https://leetcode.com/problems/match-substring-after-replacement/
Best: Brute check with replacement map — O(|s|*|sub|*|mappings|)
"""


from typing import List
from collections import defaultdict


class Solution:
    def matchReplacement(self, s: str, sub: str, mappings: List[List[str]]) -> bool:
        allow = defaultdict(set)
        for a, b in mappings:
            allow[a].add(b)
        m = len(sub)
        for i in range(len(s) - m + 1):
            ok = True
            for j in range(m):
                sc, tc = s[i + j], sub[j]
                if sc == tc or sc in allow[tc]:
                    continue
                ok = False
                break
            if ok:
                return True
        return False


def _test():
    s = Solution()
    assert s.matchReplacement("fool3e7bar", "leet", [["e","3"],["t","7"],["t","8"]]) is True
    assert s.matchReplacement("fooleetbar", "f00l", [["o","0"]]) is False
    assert s.matchReplacement("Fool33tbaR", "leetd", [["e","3"],["t","7"],["t","8"],["d","b"],["p","b"]]) is True
    print("LC 2301 OK")


if __name__ == "__main__":
    _test()

