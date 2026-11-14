"""
LC 564. Find the Closest Palindrome — Hard
https://leetcode.com/problems/find-the-closest-palindrome/
Best: mirror candidates around half prefix — O(len(n))
"""


class Solution:
    def nearestPalindromic(self, n: str) -> str:
        length = len(n)
        num = int(n)
        candidates = set()
        candidates.add(10 ** (length - 1) - 1)  # 999...9
        candidates.add(10 ** length + 1)        # 100...001

        prefix = int(n[:(length + 1) // 2])
        for p in (prefix - 1, prefix, prefix + 1):
            s = str(p)
            if length % 2 == 0:
                pal = s + s[::-1]
            else:
                pal = s + s[:-1][::-1]
            candidates.add(int(pal))

        candidates.discard(num)
        best = min(candidates, key=lambda x: (abs(x - num), x))
        return str(best)


def _test():
    s = Solution()
    assert s.nearestPalindromic("123") == "121"
    assert s.nearestPalindromic("1") == "0"
    assert s.nearestPalindromic("10") == "9"
    assert s.nearestPalindromic("11") == "9"
    assert s.nearestPalindromic("100") == "99"
    print("LC 564 OK")


if __name__ == "__main__":
    _test()
