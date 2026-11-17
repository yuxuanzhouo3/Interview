"""
LC 761. Special Binary String — Hard
Best: Recursively split specials, sort descending — O(n^2)
"""

class Solution:
    def makeLargestSpecial(self, s: str) -> str:
        if not s:
            return ""
        parts = []
        bal = i = 0
        for j, c in enumerate(s):
            bal += 1 if c == "1" else -1
            if bal == 0:
                # s[i:j+1] = 1 + mid + 0
                mid = self.makeLargestSpecial(s[i + 1:j])
                parts.append("1" + mid + "0")
                i = j + 1
        parts.sort(reverse=True)
        return "".join(parts)


def _test():
    s = Solution()
    assert s.makeLargestSpecial("11011000") == "11100100"
    assert s.makeLargestSpecial("10") == "10"
    print("LC 761 OK")


if __name__ == "__main__":
    _test()
