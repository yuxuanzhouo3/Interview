"""
LC 591. Tag Validator — Hard
https://leetcode.com/problems/tag-validator/
Best: stack parse of tags / CDATA — O(n)
"""


class Solution:
    def isValid(self, code: str) -> bool:
        n = len(code)
        stack = []
        i = 0
        while i < n:
            if i > 0 and not stack:
                return False
            if code.startswith("<![CDATA[", i):
                j = code.find("]]>", i + 9)
                if j < 0:
                    return False
                i = j + 3
            elif code.startswith("</", i):
                j = code.find(">", i + 2)
                if j < 0:
                    return False
                tag = code[i + 2 : j]
                if not stack or stack[-1] != tag:
                    return False
                stack.pop()
                i = j + 1
            elif code.startswith("<", i):
                j = code.find(">", i + 1)
                if j < 0:
                    return False
                tag = code[i + 1 : j]
                if not (1 <= len(tag) <= 9 and tag.isupper() and tag.isalpha()):
                    return False
                stack.append(tag)
                i = j + 1
            else:
                i += 1
        return not stack


def _test():
    s = Solution()
    assert s.isValid("<DIV>This is the first line <![CDATA[<div>]]></DIV>") is True
    assert s.isValid("<DIV>>>  ![cdata[]] <![CDATA[<div>]>]]>]]>>]</DIV>") is True
    assert s.isValid("<A>  <B> </A>   </B>") is False
    assert s.isValid("<A></A><B></B>") is False
    assert s.isValid("<A><A></A></A>") is True
    print("LC 591 OK")


if __name__ == "__main__":
    _test()
