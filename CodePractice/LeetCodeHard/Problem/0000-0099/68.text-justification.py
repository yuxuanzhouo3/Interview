"""
LC 68. Text Justification — Hard
Best: Greedy pack lines + distribute spaces — O(total chars)
"""

from typing import List


class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        ans = []
        i, n = 0, len(words)
        while i < n:
            j = i
            width = 0
            while j < n and width + len(words[j]) + (j - i) <= maxWidth:
                width += len(words[j])
                j += 1
            gap_slots = j - i - 1
            line_words = words[i:j]
            # last line or single word: left justify
            if j == n or gap_slots == 0:
                line = " ".join(line_words)
                line += " " * (maxWidth - len(line))
                ans.append(line)
            else:
                spaces = maxWidth - width
                base, extra = divmod(spaces, gap_slots)
                parts = []
                for k, w in enumerate(line_words):
                    parts.append(w)
                    if k < gap_slots:
                        parts.append(" " * (base + (1 if k < extra else 0)))
                ans.append("".join(parts))
            i = j
        return ans


def _test():
    s = Solution()
    r = s.fullJustify(["This", "is", "an", "example", "of", "text", "justification."], 16)
    assert r == ["This    is    an", "example  of text", "justification.  "]
    print("LC 68 OK")


if __name__ == "__main__":
    _test()
