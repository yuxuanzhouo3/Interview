"""
LC 772. Basic Calculator III — Hard
https://leetcode.com/problems/basic-calculator-iii/
Best: Recursive descent / dual stack — O(n)
"""


class Solution:
    def calculate(self, s: str) -> int:
        s = s.replace(" ", "")
        n = len(s)

        def helper(i: int):
            stack = []
            num = 0
            sign = "+"
            while i < n:
                ch = s[i]
                if ch.isdigit():
                    num = num * 10 + int(ch)
                    i += 1
                    continue
                if ch == "(":
                    num, i = helper(i + 1)
                    continue
                # operator or ')'
                if sign == "+":
                    stack.append(num)
                elif sign == "-":
                    stack.append(-num)
                elif sign == "*":
                    stack.append(stack.pop() * num)
                else:  # /
                    prev = stack.pop()
                    stack.append(int(prev / num))
                num = 0
                if ch == ")":
                    return sum(stack), i + 1
                sign = ch
                i += 1
            # end of string — flush last num
            if sign == "+":
                stack.append(num)
            elif sign == "-":
                stack.append(-num)
            elif sign == "*":
                stack.append(stack.pop() * num)
            else:
                prev = stack.pop()
                stack.append(int(prev / num))
            return sum(stack), i

        ans, _ = helper(0)
        return ans


def _test():
    s = Solution()
    assert s.calculate("1+1") == 2
    assert s.calculate("6-4/2") == 4
    assert s.calculate("2*(5+5*2)/3+(6/2+8)") == 21
    assert s.calculate("(2+6*3+5-(3*14/7+2)*5)+3") == -12
    assert s.calculate("0") == 0
    print("LC 772 OK")


if __name__ == "__main__":
    _test()
