"""
LC 3298. Count Substrings That Can Be Rearranged to Contain a String II — Hard
https://leetcode.com/problems/count-substrings-that-can-be-rearranged-to-contain-a-string-ii/
Best: Sliding window cover — O(n)
"""

from collections import Counter


class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        need = Counter(word2)
        missing = len(word2)
        cnt = Counter()
        j = 0
        ans = 0
        n = len(word1)
        for i, ch in enumerate(word1):
            cnt[ch] += 1
            if cnt[ch] <= need[ch]:
                missing -= 1
            while missing == 0:
                ans += n - i
                y = word1[j]
                if cnt[y] <= need[y]:
                    missing += 1
                cnt[y] -= 1
                j += 1
        return ans


def _test():
    s = Solution()
    assert s.validSubstringCount("bcca", "abc") == 1
    assert s.validSubstringCount("abcabc", "abc") == 10
    assert s.validSubstringCount("abcabc", "aaabc") == 0
    print("LC 3298 OK")


if __name__ == "__main__":
    _test()
