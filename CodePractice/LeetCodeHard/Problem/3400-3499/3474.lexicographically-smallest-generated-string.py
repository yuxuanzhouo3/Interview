"""
LC 3474. Lexicographically Smallest Generated String — Hard
https://leetcode.com/problems/lexicographically-smallest-generated-string/
Best: 方法一：贪心
"""

class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        s, t = str1, str2
        n, m = len(s), len(t)
        ans = ["a"] * (n + m - 1)
        fixed = [False] * (n + m - 1)

        for i, b in enumerate(s):
            if b != "T":
                continue
            for j, c in enumerate(t):
                k = i + j
                if fixed[k] and ans[k] != c:
                    return ""
                ans[k] = c
                fixed[k] = True

        for i, b in enumerate(s):
            if b != "F":
                continue
            if "".join(ans[i : i + m]) != t:
                continue
            for j in range(i + m - 1, i - 1, -1):
                if not fixed[j]:
                    ans[j] = "b"
                    break
            else:
                return ""

        return "".join(ans)

def _test():
    s = Solution()
    assert s.generateString(str1='TFTF', str2='ab') == 'ababa'
    assert s.generateString(str1='TFTF', str2='abc') == ''
    assert s.generateString(str1='F', str2='d') == 'a'
    print("LC 3474 OK")

if __name__ == "__main__":
    _test()
