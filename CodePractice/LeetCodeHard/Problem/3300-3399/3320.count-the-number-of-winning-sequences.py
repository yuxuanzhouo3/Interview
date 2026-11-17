"""
LC 3320. Count The Number of Winning Sequences — Hard
https://leetcode.com/problems/count-the-number-of-winning-sequences/
Best: DP on score diff + last move — O(n^2)
"""

MOD = 10**9 + 7


class Solution:
    def countWinningSequences(self, s: str) -> int:
        # FBE beats: F>E, W>F, E>W (Fire/Water/Earth) — actually problem uses F,W,E
        # Alice sequence s; Bob chooses sequence with no two consecutive same.
        # Bob wins if score > Alice. Count Bob winning sequences.
        mp = {"F": 0, "W": 1, "E": 2}

        def beat(a, b):
            # a beats b?
            return (a - b) % 3 == 1

        n = len(s)
        alice = [mp[c] for c in s]
        # dp[i][diff][last] after i rounds; diff offset by n
        # optimize: iterate rounds
        offset = n
        # use dict or array: diff in [-n,n]
        from collections import defaultdict

        # dp[last][diff] 
        dp = [defaultdict(int) for _ in range(3)]
        for b in range(3):
            d = 0
            if beat(b, alice[0]):
                d = 1
            elif beat(alice[0], b):
                d = -1
            dp[b][d] = 1
        for i in range(1, n):
            ndp = [defaultdict(int) for _ in range(3)]
            a = alice[i]
            for last in range(3):
                for diff, ways in dp[last].items():
                    for b in range(3):
                        if b == last:
                            continue
                        nd = diff
                        if beat(b, a):
                            nd += 1
                        elif beat(a, b):
                            nd -= 1
                        ndp[b][nd] = (ndp[b][nd] + ways) % MOD
            dp = ndp
        ans = 0
        for last in range(3):
            for diff, ways in dp[last].items():
                if diff > 0:
                    ans = (ans + ways) % MOD
        return ans


def _test():
    s = Solution()
    assert s.countWinningSequences("FFF") == 3
    assert s.countWinningSequences("FWEFW") == 18
    print("LC 3320 OK")


if __name__ == "__main__":
    _test()
