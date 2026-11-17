"""
LC 1987. Number of Unique Good Subsequences — Hard
https://leetcode.com/problems/number-of-unique-good-subsequences/
Best: DP endswith0/1 unique binary subsequences — O(n)
"""

MOD = 10**9 + 7


class Solution:
    def numberOfUniqueGoodSubsequences(self, binary: str) -> int:
        # end0 / end1: # distinct good subsequences ending with 0 / 1
        end0 = end1 = 0
        has_zero = False
        for ch in binary:
            if ch == "1":
                # append 1 to any existing good subseq, or start new "1"
                end1 = (end0 + end1 + 1) % MOD
            else:
                # append 0 to subsequences ending with 1 or 0 (that already started with 1)
                end0 = (end0 + end1) % MOD
                has_zero = True
        ans = (end0 + end1) % MOD
        if has_zero:
            ans = (ans + 1) % MOD  # subsequence "0"
        return ans


def _test():
    s = Solution()
    assert s.numberOfUniqueGoodSubsequences("001") == 2
    assert s.numberOfUniqueGoodSubsequences("11") == 2
    assert s.numberOfUniqueGoodSubsequences("101") == 5
    print("LC 1987 OK")


if __name__ == "__main__":
    _test()
