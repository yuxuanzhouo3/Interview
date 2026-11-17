"""
LC 3003. Maximize the Number of Partitions After Operations — Hard
https://leetcode.com/problems/maximize-the-number-of-partitions-after-operations/
Best: Bitmask DP partitions with one change — O(n * 2^10)
"""

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        # greedy partitions without change
        def count_parts(string):
            ans = 0
            i = 0
            while i < len(string):
                mask = 0
                while i < len(string):
                    bit = 1 << (ord(string[i]) - 97)
                    if bin(mask | bit).count("1") > k:
                        break
                    mask |= bit
                    i += 1
                ans += 1
            return ans

        # With at most one index change: try change each position (n<=10000, need smarter)
        # Official: DP state (i, can_change, mask) memo
        from functools import lru_cache

        @lru_cache(None)
        def dfs(i: int, changed: int, mask: int) -> int:
            if i == n:
                return 0
            ans = 0
            bit = 1 << (ord(s[i]) - 97)
            nmask = mask | bit
            if bin(nmask).count("1") > k:
                ans = max(ans, 1 + dfs(i + 1, changed, bit))
            else:
                ans = max(ans, dfs(i + 1, changed, nmask))
            if not changed:
                for c in range(26):
                    b = 1 << c
                    nm = mask | b
                    if bin(nm).count("1") > k:
                        ans = max(ans, 1 + dfs(i + 1, 1, b))
                    else:
                        ans = max(ans, dfs(i + 1, 1, nm))
            return ans

        return dfs(0, 0, 0) + (1 if n else 0)


def _test():
    s = Solution()
    assert s.maxPartitionsAfterOperations("accca", 2) == 3
    assert s.maxPartitionsAfterOperations("aabaab", 3) == 1
    assert s.maxPartitionsAfterOperations("xxyz", 1) == 4
    print("LC 3003 OK")


if __name__ == "__main__":
    _test()

