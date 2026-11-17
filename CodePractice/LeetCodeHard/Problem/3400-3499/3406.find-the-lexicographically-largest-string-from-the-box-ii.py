"""
LC 3406. Find the Lexicographically Largest String From the Box II — Hard
https://leetcode.com/problems/find-the-lexicographically-largest-string-from-the-box-ii/
Best: Last substring (Duval-like) + truncate — O(n)
"""


class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        s = self._last_substring(word)
        sz = len(word) - numFriends + 1
        return s[: min(len(s), sz)]

    def _last_substring(self, s: str) -> str:
        i = j = k = 0
        j = 1
        while j + k < len(s):
            if s[i + k] == s[j + k]:
                k += 1
            elif s[i + k] > s[j + k]:
                j = j + k + 1
                k = 0
            else:
                i = max(i + k + 1, j)
                j = i + 1
                k = 0
        return s[i:]


def _test():
    s = Solution()
    # same as Box I examples when applicable
    assert s.answerString("dbca", 2) == "dbc"
    assert s.answerString("gggg", 4) == "g"
    assert s.answerString("aann", 1) == "aann"
    print("LC 3406 OK")


if __name__ == "__main__":
    _test()
