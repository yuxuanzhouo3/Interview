"""
LC 301. Remove Invalid Parentheses — Hard
Best: BFS level by level remove — shortest removals
"""

from typing import List


class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def valid(t):
            bal = 0
            for c in t:
                if c == "(":
                    bal += 1
                elif c == ")":
                    bal -= 1
                    if bal < 0:
                        return False
            return bal == 0

        level = {s}
        while True:
            good = [x for x in level if valid(x)]
            if good:
                return good
            nxt = set()
            for x in level:
                for i in range(len(x)):
                    if x[i] in "()":
                        nxt.add(x[:i] + x[i + 1:])
            level = nxt


def _test():
    s = Solution()
    assert sorted(s.removeInvalidParentheses("()())()")) == sorted(["(())()", "()()()"])
    print("LC 301 OK")


if __name__ == "__main__":
    _test()
