"""
LC 2223. Sum of Scores of Built Strings — Hard
https://leetcode.com/problems/sum-of-scores-of-built-strings/
Best: Z-algorithm sum of LCP with prefix — O(n)
"""


class Solution:
    def sumScores(self, s: str) -> int:
        n = len(s)
        z = [0] * n
        l = r = 0
        for i in range(1, n):
            if i <= r:
                z[i] = min(r - i + 1, z[i - l])
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1
        z[0] = n
        return sum(z)


def _test():
    s = Solution()
    assert s.sumScores("babab") == 9
    assert s.sumScores("azbazbzaz") == 14
    print("LC 2223 OK")


if __name__ == "__main__":
    _test()

