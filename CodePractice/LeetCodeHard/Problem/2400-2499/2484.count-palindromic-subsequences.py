"""
LC 2484. Count Palindromic Subsequences — Hard
https://leetcode.com/problems/count-palindromic-subsequences/
Best: Count 5-digit palindrome subsequences by digit pairs — O(n*100)
"""


class Solution:
    def countPalindromes(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # pref[i][d] = count of d in s[:i]
        pref = [[0] * 10 for _ in range(n + 1)]
        for i, ch in enumerate(s):
            pref[i + 1] = pref[i][:]
            pref[i + 1][ord(ch) - 48] += 1
        suf = [[0] * 10 for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            suf[i] = suf[i + 1][:]
            suf[i][ord(s[i]) - 48] += 1

        # for each middle index, count patterns abXba
        # pre_pair[a][b] before i, suf_pair[a][b] after i
        ans = 0
        # pre_ab[a][b]: number of subsequences "ab" ending before current
        pre_ab = [[0] * 10 for _ in range(10)]
        # cnt_left[d]
        cnt_l = [0] * 10
        # we need also suf_ab built from right
        # Process left to right for middle, maintaining left pairs; right pairs from precomputed
        # Build all right pair counts from each position
        # suf_ab_at[i][a][b] = number of "ab" subsequences in s[i+1:]  where a then b later... 
        # For abXba: left needs a then b before X; right needs b then a after X.
        # So right: count pairs (b,a) in order after X.

        suf_ba = [[0] * 10 for _ in range(10)]  # will maintain from right: pairs (first,second)=(b,a)
        cnt_r = [0] * 10
        # Actually iterate middle from left to right: first build full right, then remove as middle moves
        for i in range(n - 1, -1, -1):
            d = ord(s[i]) - 48
            # before adding s[i], existing cnt_r are to the right of i
            # When we process middles later we need different state - use two-pass differently

        # Clear approach:
        # For fixed middle m, 
        # left: for each (a,b) count ways a..b before m
        # right: for each (b,a) count ways b..a after m
        # Precompute left_ab[m][a][b] and right_ba[m][b][a]

        left_ab = [[[0]*10 for _ in range(10)] for _ in range(n)]
        cnt = [0]*10
        ab = [[0]*10 for _ in range(10)]
        for m in range(n):
            for a in range(10):
                for b in range(10):
                    left_ab[m][a][b] = ab[a][b]
            # after recording for middle m, add s[m] into left for future
            d = ord(s[m]) - 48
            for a in range(10):
                ab[a][d] += cnt[a]
            cnt[d] += 1

        right_ba = [[[0]*10 for _ in range(10)] for _ in range(n)]
        cnt = [0]*10
        ba = [[0]*10 for _ in range(10)]  # pair (b,a): first b then a going leftward... 
        # building from right: when scanning right to left, "ba" means b appears then a further right
        # so as we go right-to-left, when at position, pairs already in ba are to the right
        cnt = [0]*10
        pair = [[0]*10 for _ in range(10)]  # pair[x][y] = x then y to the right (x closer to middle)
        for m in range(n - 1, -1, -1):
            for b in range(10):
                for a in range(10):
                    right_ba[m][b][a] = pair[b][a]
            d = ord(s[m]) - 48
            for y in range(10):
                # d then y (y already counted on right)
                pair[d][y] += cnt[y]
            cnt[d] += 1

        for m in range(n):
            for a in range(10):
                for b in range(10):
                    ans = (ans + left_ab[m][a][b] * right_ba[m][b][a]) % MOD
        return ans


def _test():
    s = Solution()
    assert s.countPalindromes("103301") == 2
    assert s.countPalindromes("0000000") == 21
    assert s.countPalindromes("9999900000") == 2
    print("LC 2484 OK")


if __name__ == "__main__":
    _test()

