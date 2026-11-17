"""
LC 3518. Smallest Palindromic Rearrangement II — Hard
https://leetcode.com/problems/smallest-palindromic-rearrangement-ii/
Best: 方法一
"""

class Solution:
    def smallestPalindrome(self, s: str, k: int) -> str:
        limit = 1_000_001
        freq = [0] * 26
        for ch in s:
            freq[ord(ch) - 97] += 1
        odd = 0
        mid = 26
        for i, v in enumerate(freq):
            if v & 1:
                odd += 1
                mid = i
        if odd > 1:
            return ""
        half = [v // 2 for v in freq]
        half_len = sum(half)

        def count_permutations(counts: list[int]) -> int:
            remaining = sum(counts)
            perms = 1
            for count in counts:
                if count == 0:
                    continue
                selected = min(count, remaining - count)
                combos = 1
                for step in range(1, selected + 1):
                    combos = combos * (remaining - step + 1) // step
                    if combos >= limit:
                        combos = limit
                        break
                perms *= combos
                if perms >= limit:
                    return limit
                remaining -= count
            return perms

        if k > count_permutations(half):
            return ""
        n = len(s)
        pal = [""] * n
        rank = k
        pos = 0
        for _ in range(half_len):
            for i in range(26):
                if half[i] == 0:
                    continue
                half[i] -= 1
                suffix = count_permutations(half)
                if suffix >= rank:
                    pal[pos] = chr(97 + i)
                    pos += 1
                    break
                rank -= suffix
                half[i] += 1
        if mid < 26:
            pal[half_len] = chr(97 + mid)
        for i in range(half_len):
            pal[n - 1 - i] = pal[i]
        return "".join(pal)

def _test():
    s = Solution()
    assert s.smallestPalindrome(s='abba', k=2) == 'baab'
    assert s.smallestPalindrome(s='aa', k=2) == ''
    assert s.smallestPalindrome(s='bacab', k=1) == 'abcba'
    print("LC 3518 OK")

if __name__ == "__main__":
    _test()
