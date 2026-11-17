"""
LC 2156. Find Substring With Given Hash Value — Hard
https://leetcode.com/problems/find-substring-with-given-hash-value/
Best: rolling hash from right — O(n)
"""


class Solution:
    def subStrHash(self, s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
        n = len(s)

        def val(c: str) -> int:
            return ord(c) - 96  # 'a' -> 1

        h = 0
        pk = 1
        for j in range(k):
            h = (h + val(s[n - k + j]) * pk) % modulo
            if j + 1 < k:
                pk = pk * power % modulo

        start = n - k if h == hashValue else -1
        for i in range(n - k - 1, -1, -1):
            h = (h - val(s[i + k]) * pk) % modulo
            h = (h * power + val(s[i])) % modulo
            if h == hashValue:
                start = i
        return s[start : start + k]


def _test():
    s = Solution()
    assert s.subStrHash("leetcode", 7, 20, 2, 0) == "ee"
    assert s.subStrHash("fbxzaad", 31, 100, 3, 32) == "fbx"
    print("LC 2156 OK")


if __name__ == "__main__":
    _test()
