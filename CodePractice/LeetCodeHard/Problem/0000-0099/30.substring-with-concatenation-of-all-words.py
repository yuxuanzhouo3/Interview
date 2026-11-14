"""
LC 30. Substring with Concatenation of All Words — Hard
Best: Sliding window by word length — O(n * wordLen)
"""

from typing import List
from collections import Counter


class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words:
            return []
        wlen = len(words[0])
        nwords = len(words)
        total = wlen * nwords
        need = Counter(words)
        ans = []
        for offset in range(wlen):
            left = offset
            window = Counter()
            count = 0
            for right in range(offset, len(s) - wlen + 1, wlen):
                w = s[right:right + wlen]
                window[w] += 1
                count += 1
                while window[w] > need[w]:
                    left_w = s[left:left + wlen]
                    window[left_w] -= 1
                    left += wlen
                    count -= 1
                if count == nwords:
                    ans.append(left)
        return ans


def _test():
    s = Solution()
    assert s.findSubstring("barfoothefoobarman", ["foo", "bar"]) == [0, 9]
    assert s.findSubstring("wordgoodgoodgoodbestword", ["word", "good", "best", "word"]) == []
    print("LC 30 OK")


if __name__ == "__main__":
    _test()
