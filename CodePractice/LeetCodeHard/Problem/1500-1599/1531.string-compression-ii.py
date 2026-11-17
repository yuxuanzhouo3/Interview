"""
LC 1531. String Compression II — Hard
https://leetcode.com/problems/string-compression-ii/
Best: DP delete ≤k — O(n^2 · k)
"""


class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        n = len(s)
        INF = 10**9
        # dp[i][left] = min encoded length for s[i:] with `left` deletions
        from functools import lru_cache

        def encoded_len(cnt: int) -> int:
            if cnt == 0:
                return 0
            if cnt == 1:
                return 1
            if cnt < 10:
                return 2
            if cnt < 100:
                return 3
            return 4

        @lru_cache(None)
        def dfs(i: int, left: int) -> int:
            if n - i <= left:
                return 0
            # delete s[i]
            ans = dfs(i + 1, left - 1) if left else INF
            # keep s[i] as start of a run; delete some later different chars
            keep = 0
            del_used = 0
            for j in range(i, n):
                if s[j] == s[i]:
                    keep += 1
                    ans = min(ans, encoded_len(keep) + dfs(j + 1, left - del_used))
                else:
                    del_used += 1
                    if del_used > left:
                        break
            return ans

        return dfs(0, k)


def _test():
    s = Solution()
    assert s.getLengthOfOptimalCompression("aaabcccd", 2) == 4
    assert s.getLengthOfOptimalCompression("aabbaa", 2) == 2
    assert s.getLengthOfOptimalCompression("aaaaaaaaaaa", 0) == 3
    print("LC 1531 OK")


if __name__ == "__main__":
    _test()
