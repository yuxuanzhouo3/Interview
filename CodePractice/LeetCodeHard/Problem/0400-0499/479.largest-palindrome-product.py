"""
LC 479. Largest Palindrome Product — Hard
https://leetcode.com/problems/largest-palindrome-product/
Best: Enumerate palindrome from n-digit products — O(10^n)
"""


class Solution:
    def largestPalindrome(self, n: int) -> int:
        if n == 1:
            return 9
        upper = 10**n - 1
        lower = 10 ** (n - 1)
        for left in range(upper, lower - 1, -1):
            # build palindrome: left + reverse(left)
            s = str(left)
            pal = int(s + s[::-1])
            # check if pal = a * b with both n-digit
            a = upper
            while a * a >= pal:
                if pal % a == 0:
                    b = pal // a
                    if lower <= b <= upper:
                        return pal % 1337
                a -= 1
        return 0


def _test():
    s = Solution()
    assert s.largestPalindrome(1) == 9
    assert s.largestPalindrome(2) == 987  # 993*913 = 906609 % 1337
    assert s.largestPalindrome(3) == 123  # known answer
    print("LC 479 OK")


if __name__ == "__main__":
    _test()
