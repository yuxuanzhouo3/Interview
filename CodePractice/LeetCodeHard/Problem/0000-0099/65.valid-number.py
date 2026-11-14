"""
LC 65. Valid Number — Hard
Best: DFA / state machine scan — O(n)
"""

class Solution:
    def isNumber(self, s: str) -> bool:
        s = s.strip()
        num = exp = dot = False
        for i, c in enumerate(s):
            if c in "eE":
                if exp or not num:
                    return False
                exp = True
                num = False
            elif c in "+-":
                if i > 0 and s[i - 1] not in "eE":
                    return False
            elif c == ".":
                if dot or exp:
                    return False
                dot = True
            elif c.isdigit():
                num = True
            else:
                return False
        return num


def _test():
    s = Solution()
    assert s.isNumber("0") is True
    assert s.isNumber("e") is False
    assert s.isNumber(".") is False
    assert s.isNumber("2e10") is True
    assert s.isNumber("-90E3") is True
    assert s.isNumber("95a54e53") is False
    print("LC 65 OK")


if __name__ == "__main__":
    _test()
