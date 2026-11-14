"""
LC 336. Palindrome Pairs — Hard
Best: HashMap word→index + prefix/suffix palindrome check — O(n * L^2)
"""

from typing import List


class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        idx = {w: i for i, w in enumerate(words)}
        ans = []

        def is_pal(s, l, r):
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        for i, w in enumerate(words):
            n = len(w)
            for cut in range(n + 1):
                # left palindrome → need reverse(right)
                if is_pal(w, 0, cut - 1):
                    rev = w[cut:][::-1]
                    j = idx.get(rev)
                    if j is not None and j != i:
                        ans.append([j, i])
                # right palindrome → need reverse(left); skip cut==n to avoid dup empty
                if cut < n and is_pal(w, cut, n - 1):
                    rev = w[:cut][::-1]
                    j = idx.get(rev)
                    if j is not None and j != i:
                        ans.append([i, j])
        return ans


def _test():
    s = Solution()
    r = s.palindromePairs(["abcd", "dcba", "lls", "s", "sssll"])
    assert sorted(r) == sorted([[0, 1], [1, 0], [3, 2], [2, 4]])
    print("LC 336 OK")


if __name__ == "__main__":
    _test()
