"""
LC 3260. Find the Largest Palindrome Divisible by K — Hard
https://leetcode.com/problems/find-the-largest-palindrome-divisible-by-k/
Best: Case by k construction / DP digits — O(n) or O(1) patterns
"""


class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if k in (1, 3, 9):
            return "9" * n
        if k == 2:
            return "8" * n if n <= 2 else "8" + "9" * (n - 2) + "8"
        if k == 4:
            return "8" * n if n <= 4 else "88" + "9" * (n - 4) + "88"
        if k == 5:
            return "5" * n if n <= 2 else "5" + "9" * (n - 2) + "5"
        if k == 8:
            return "8" * n if n <= 6 else "888" + "9" * (n - 6) + "888"
        if k == 6:
            if n <= 2:
                return "6" * n
            if n % 2:
                l = n // 2 - 1
                return "8" + "9" * l + "8" + "9" * l + "8"
            l = n // 2 - 2
            return "8" + "9" * l + "77" + "9" * l + "8"
        # k == 7: period 6 for 10^i mod 7
        middle = {
            0: "",
            1: "7",
            2: "77",
            3: "959",
            4: "9779",
            5: "99799",
            6: "999999",
            7: "9994999",
            8: "99944999",
            9: "999969999",
            10: "9999449999",
            11: "99999499999",
        }
        q, r = divmod(n, 12)
        return "999999" * q + middle[r] + "999999" * q


def _test():
    s = Solution()
    assert s.largestPalindrome(3, 5) == "595"
    assert s.largestPalindrome(1, 4) == "8"
    assert s.largestPalindrome(5, 6) == "89898"
    for n in range(1, 20):
        for k in range(1, 10):
            pal = s.largestPalindrome(n, k)
            assert pal == pal[::-1] and len(pal) == n and int(pal) % k == 0
    print("LC 3260 OK")


if __name__ == "__main__":
    _test()
