"""
LC 1096. Brace Expansion II — Hard
https://leetcode.com/problems/brace-expansion-ii/
Best: Recursive descent / stack union-product — O(R)
"""

from typing import List


class Solution:
    def braceExpansionII(self, expression: str) -> List[str]:
        def parse(i):
            # returns (set of strings, next index)
            union = set()
            cur = {""}
            n = len(expression)
            while i < n:
                c = expression[i]
                if c == "{":
                    inner, i = parse(i + 1)
                    cur = {a + b for a in cur for b in inner}
                elif c == "}":
                    union |= cur
                    return union, i + 1
                elif c == ",":
                    union |= cur
                    cur = {""}
                    i += 1
                else:
                    j = i
                    while j < n and expression[j].isalpha():
                        j += 1
                    word = expression[i:j]
                    cur = {a + word for a in cur}
                    i = j
            union |= cur
            return union, i

        res, _ = parse(0)
        return sorted(res)


def _test():
    s = Solution()
    assert s.braceExpansionII("{a,b}{c,{d,e}}") == ["ac", "ad", "ae", "bc", "bd", "be"]
    assert s.braceExpansionII("{{a,z},a{b,c},{ab,z}}") == ["a", "ab", "ac", "z"]
    assert s.braceExpansionII("{a,b}") == ["a", "b"]
    print("LC 1096 OK")


if __name__ == "__main__":
    _test()
