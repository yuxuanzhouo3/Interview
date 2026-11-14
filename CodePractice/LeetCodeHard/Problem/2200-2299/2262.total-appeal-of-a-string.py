"""
LC 2262. Total Appeal of A String — Hard
https://leetcode.com/problems/total-appeal-of-a-string/
Best: Last-seen contribution — O(n)
"""

class Solution:
    def appealSum(self, s: str) -> int:
        last = [-1] * 26
        ans = 0
        for i, ch in enumerate(s):
            c = ord(ch) - 97
            ans += (i - last[c]) * (len(s) - i)
            last[c] = i
        return ans


def _test():
    s = Solution()
    assert s.appealSum("abbca") == 28
    assert s.appealSum("code") == 20
    print("LC 2262 OK")


if __name__ == "__main__":
    _test()
