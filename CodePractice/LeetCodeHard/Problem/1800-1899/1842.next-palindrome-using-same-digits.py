"""
LC 1842. Next Palindrome Using Same Digits — Hard
https://leetcode.com/problems/next-palindrome-using-same-digits/
Best: 对左半做 next_permutation 再镜像 — O(n)
"""


class Solution:
    def nextPalindrome(self, num: str) -> str:
        n = len(num)
        half = list(num[: n // 2])
        if not self._next_perm(half):
            return ""
        left = "".join(half)
        mid = num[n // 2] if n % 2 else ""
        return left + mid + left[::-1]

    def _next_perm(self, a: list) -> bool:
        i = len(a) - 2
        while i >= 0 and a[i] >= a[i + 1]:
            i -= 1
        if i < 0:
            return False
        j = len(a) - 1
        while a[j] <= a[i]:
            j -= 1
        a[i], a[j] = a[j], a[i]
        a[i + 1 :] = reversed(a[i + 1 :])
        return True


def _test():
    s = Solution()
    assert s.nextPalindrome("1221") == "2112"
    assert s.nextPalindrome("32123") == ""
    assert s.nextPalindrome("45544554") == "54455445"
    print("LC 1842 OK")


if __name__ == "__main__":
    _test()
