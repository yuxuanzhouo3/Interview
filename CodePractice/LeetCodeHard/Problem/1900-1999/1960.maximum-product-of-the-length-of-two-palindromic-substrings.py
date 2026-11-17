"""
LC 1960. Maximum Product of the Length of Two Palindromic Substrings — Hard
https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/
Best: Manacher + prefix/suffix max odd palindrome — O(n)
"""

class Solution:
    def maxProduct(self, s: str) -> int:
        n = len(s)
        hlen = [0] * n
        center = right = 0
        for i in range(n):
            if i < right:
                hlen[i] = min(right - i, hlen[2 * center - i])
            while 0 <= i - 1 - hlen[i] and i + 1 + hlen[i] < n and s[i - 1 - hlen[i]] == s[i + 1 + hlen[i]]:
                hlen[i] += 1
            if right < i + hlen[i]:
                center, right = i, i + hlen[i]

        prefix = [0] * n
        suffix = [0] * n
        for i in range(n):
            prefix[i + hlen[i]] = max(prefix[i + hlen[i]], 2 * hlen[i] + 1)
            suffix[i - hlen[i]] = max(suffix[i - hlen[i]], 2 * hlen[i] + 1)
        for i in range(1, n):
            prefix[~i] = max(prefix[~i], prefix[~i + 1] - 2)
            suffix[i] = max(suffix[i], suffix[i - 1] - 2)
        for i in range(1, n):
            prefix[i] = max(prefix[i - 1], prefix[i])
            suffix[~i] = max(suffix[~i], suffix[~i + 1])
        return max(prefix[i - 1] * suffix[i] for i in range(1, n))

def _test():

    s = Solution()
    assert s.maxProduct("ababbb") == 9
    assert s.maxProduct("zaaaxbbby") == 9
    print("LC 1960 OK")

if __name__ == "__main__":
    _test()
