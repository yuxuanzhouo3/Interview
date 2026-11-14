"""
LC 224. Basic Calculator — Hard
Best: Stack for signs / parentheses — O(n)
"""

class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        num = 0
        sign = 1
        ans = 0
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in "+-":
                ans += sign * num
                num = 0
                sign = 1 if c == "+" else -1
            elif c == "(":
                stack.append(ans)
                stack.append(sign)
                ans = 0
                sign = 1
            elif c == ")":
                ans += sign * num
                num = 0
                ans *= stack.pop()
                ans += stack.pop()
        return ans + sign * num


def _test():
    s = Solution()
    assert s.calculate("1 + 1") == 2
    assert s.calculate(" 2-1 + 2 ") == 3
    assert s.calculate("(1+(4+5+2)-3)+(6+8)") == 23
    print("LC 224 OK")


if __name__ == "__main__":
    _test()
