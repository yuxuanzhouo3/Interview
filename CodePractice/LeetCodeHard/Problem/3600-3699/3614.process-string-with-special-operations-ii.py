"""
LC 3614. Process String with Special Operations II — Hard
https://leetcode.com/problems/process-string-with-special-operations-ii/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def processStr(self, s: str, k: int) -> str:
        m = 0
        for c in s:
            if c == "*":
                m = max(0, m - 1)
            elif c == "#":
                m <<= 1
            elif c != "%":
                m += 1
        if k >= m:
            return "."
        for c in reversed(s):
            if c == "*":
                m += 1
            elif c == "#":
                m //= 2
                if k >= m:
                    k -= m
            elif c == "%":
                k = m - 1 - k
            else:
                m -= 1
                if k == m:
                    return c

def _test():
    s = Solution()
    assert s.processStr("a#b%*", 1) == "a"
    assert s.processStr("cd%#*#", 3) == "d"
    assert s.processStr("z*#", 0) == "."
    print("LC 3614 OK")


if __name__ == "__main__":
    _test()
