"""
LC 1147. Longest Chunked Palindrome Decomposition — Hard
https://leetcode.com/problems/longest-chunked-palindrome-decomposition/
Best: Greedy two pointers — O(n^2) / rolling hash O(n)
"""

class Solution:
    def longestDecomposition(self, text: str) -> int:
        n = len(text)
        ans = 0
        i, j = 0, n
        while i < j:
            found = False
            for k in range(1, (j - i) // 2 + 1):
                if text[i : i + k] == text[j - k : j]:
                    ans += 2
                    i += k
                    j -= k
                    found = True
                    break
            if not found:
                ans += 1
                break
        return ans


def _test():
    s = Solution()
    assert s.longestDecomposition("ghiabcdefhelloadamhelloabcdefghi") == 7
    assert s.longestDecomposition("merchant") == 1
    assert s.longestDecomposition("antaprezatepzapreanta") == 11
    print("LC 1147 OK")


if __name__ == "__main__":
    _test()
