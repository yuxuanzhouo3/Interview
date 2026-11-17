"""
LC 1320. Minimum Distance to Type a Word Using Two Fingers — Hard
https://leetcode.com/problems/minimum-distance-to-type-a-word-using-two-fingers/
Best: DP on finger positions — O(26^2 · n) → optimize O(26·n)
"""


class Solution:
    def minimumDistance(self, word: str) -> int:
        def dist(a: int, b: int) -> int:
            if a < 0 or b < 0:
                return 0
            return abs(a // 6 - b // 6) + abs(a % 6 - b % 6)

        # dp[i][a][b] too big; use: after typing word[:i], one finger on last char, other on j
        # Optimize: dp[j] = min cost with other finger on j, last typed by "free" finger at word[i-1]
        n = len(word)
        pos = [ord(c) - 65 for c in word]
        INF = 10**9
        # dp[other] = min cost; current finger is on previous letter
        dp = [0] * 26  # initially other finger nowhere: use -1 encoded as all same
        # Better approach: 3D compressed
        # f[i][j] = min cost to type first i letters with one finger on word[i-1], other on j (0..25), j=26 means nowhere
        # Use rolling
        NONE = 26
        cur = [[INF] * 27 for _ in range(27)]
        cur[NONE][NONE] = 0
        for ch in pos:
            nxt = [[INF] * 27 for _ in range(27)]
            for a in range(27):
                for b in range(27):
                    if cur[a][b] >= INF:
                        continue
                    # move finger a
                    nxt[ch][b] = min(nxt[ch][b], cur[a][b] + dist(a if a < 26 else -1, ch))
                    # move finger b
                    nxt[a][ch] = min(nxt[a][ch], cur[a][b] + dist(b if b < 26 else -1, ch))
            cur = nxt
        return min(min(row) for row in cur)


def _test():
    s = Solution()
    assert s.minimumDistance("CAKE") == 3
    assert s.minimumDistance("HAPPY") == 6
    assert s.minimumDistance("NEW") == 3
    assert s.minimumDistance("YEAR") == 7
    print("LC 1320 OK")


if __name__ == "__main__":
    _test()
