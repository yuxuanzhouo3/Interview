"""
LC 3999. Minimum Number of String Groups Through Transformations — Hard
https://leetcode.com/problems/minimum-number-of-string-groups-through-transformations/
Best: canonicalize even/odd subsequences by minimal cyclic rotation
"""


class Solution:
    def minimumGroups(self, words: list[str]) -> int:
        def min_rotation(s: str) -> str:
            if not s:
                return s
            n = len(s)
            ss = s + s
            fail = [-1] * (2 * n)
            res = 0
            j = 1
            while j < 2 * n:
                i = fail[j - res - 1]
                while i != -1 and ss[j] != ss[res + i + 1]:
                    if ss[j] < ss[res + i + 1]:
                        res = j - i - 1
                    i = fail[i]
                if i == -1 and ss[j] != ss[res]:
                    if ss[j] < ss[res]:
                        res = j
                    fail[j - res] = -1
                else:
                    fail[j - res] = i + 1
                if res >= n:
                    break
                j += 1
            res = res if res < n else res - n
            return s[res:] + s[:res]

        seen = set()
        for w in words:
            even = "".join(w[i] for i in range(0, len(w), 2))
            odd = "".join(w[i] for i in range(1, len(w), 2))
            seen.add(min_rotation(even) + "#" + min_rotation(odd))
        return len(seen)


def _test():
    s = Solution()
    assert s.minimumGroups(["ntgwz", "zwntg"]) == 1
    assert s.minimumGroups(["abc", "cab", "bac", "acb", "bca", "cba"]) == 3
    assert s.minimumGroups(["leet", "abb", "bab", "deed", "edde", "code", "bba"]) == 5
    print("LC 3999 OK")


if __name__ == "__main__":
    _test()
