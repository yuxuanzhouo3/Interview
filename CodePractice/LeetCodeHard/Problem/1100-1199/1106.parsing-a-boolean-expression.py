"""
LC 1106. Parsing A Boolean Expression — Hard
https://leetcode.com/problems/parsing-a-boolean-expression/
Best: Recursive parse / stack — O(n)
"""

class Solution:
    def parseBoolExpr(self, expression: str) -> bool:
        def parse(i):
            c = expression[i]
            if c == "t":
                return True, i + 1
            if c == "f":
                return False, i + 1
            # c in !&|
            op = c
            # skip op and '('
            i += 2
            vals = []
            while expression[i] != ")":
                if expression[i] == ",":
                    i += 1
                    continue
                v, i = parse(i)
                vals.append(v)
            i += 1  # ')'
            if op == "!":
                return (not vals[0]), i
            if op == "&":
                return all(vals), i
            return any(vals), i

        ans, _ = parse(0)
        return ans


def _test():
    s = Solution()
    assert s.parseBoolExpr("&(|(f))") is False
    assert s.parseBoolExpr("|(f,f,f,t)") is True
    assert s.parseBoolExpr("!(&(f,t))") is True
    print("LC 1106 OK")


if __name__ == "__main__":
    _test()
