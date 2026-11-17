"""
LC 3088. Make String Anti-palindrome — Hard
https://leetcode.com/problems/make-string-anti-palindrome/
Best: Count sort + greedy swap right half — O(n)
"""


class Solution:
    def makeAntiPalindrome(self, s: str) -> str:
        n = len(s)
        cnt = [0] * 26
        for c in s:
            cnt[ord(c) - 97] += 1
        if max(cnt) > n // 2:
            return "-1"
        arr = []
        for i, c in enumerate(cnt):
            arr.extend([chr(97 + i)] * c)
        half = n // 2
        i = half
        while i < n and arr[i] == arr[n - 1 - i]:
            j = i
            while j < n and arr[j] == arr[i]:
                j += 1
            if j == n:
                return "-1"
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
        return "".join(arr)


def _test():
    s = Solution()
    assert s.makeAntiPalindrome("abca") == "aabc"
    r = s.makeAntiPalindrome("abba")
    assert r == "aabb"
    assert all(r[i] != r[len(r) - 1 - i] for i in range(len(r) // 2))
    assert s.makeAntiPalindrome("aaab") == "-1"
    print("LC 3088 OK")


if __name__ == "__main__":
    _test()
