"""
LC 2953. Count Complete Substrings — Hard
https://leetcode.com/problems/count-complete-substrings/
Best: Split by adjacent diff>2 + window of c*k distinct — O(26 n)
"""


class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def count_seg(seg: str) -> int:
            n = len(seg)
            ans = 0
            for c in range(1, 27):
                L = c * k
                if L > n:
                    break
                freq = [0] * 26
                distinct = 0
                exact = 0  # how many letters have count == k

                def add(ch, delta):
                    nonlocal distinct, exact
                    i = ord(ch) - 97
                    if freq[i] == k:
                        exact -= 1
                    if freq[i] == 0 and delta == 1:
                        distinct += 1
                    if freq[i] == 1 and delta == -1:
                        distinct -= 1
                    freq[i] += delta
                    if freq[i] == k:
                        exact += 1

                for i, ch in enumerate(seg):
                    add(ch, 1)
                    if i >= L:
                        add(seg[i - L], -1)
                    if i >= L - 1 and distinct == c and exact == c:
                        ans += 1
            return ans

        ans = 0
        i = 0
        n = len(word)
        while i < n:
            j = i + 1
            while j < n and abs(ord(word[j]) - ord(word[j - 1])) <= 2:
                j += 1
            ans += count_seg(word[i:j])
            i = j
        return ans


def _test():
    s = Solution()
    assert s.countCompleteSubstrings("igigee", 2) == 3
    assert s.countCompleteSubstrings("aaabbbccc", 3) == 6
    print("LC 2953 OK")


if __name__ == "__main__":
    _test()
