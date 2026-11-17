"""
LC 3352. Count K-Reducible Numbers Less Than N — Hard
https://leetcode.com/problems/count-k-reducible-numbers-less-than-n/
Best: Digit DP + popcount reduction — O(|n| * |n| * k)
"""

from functools import lru_cache


class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # precompute steps for popcount values
        # x -> popcount(x) until 1; reducible if <= k steps from the number itself:
        # operation: replace n with popcount(n). Count numbers < s that become 1 in <= k ops.
        max_bits = n
        steps = [0] * (max_bits + 1)
        for i in range(2, max_bits + 1):
            steps[i] = 1 + steps[bin(i).count("1")]

        @lru_cache(None)
        def dp(pos: int, tight: int, ones: int, started: int) -> int:
            if pos == n:
                if not started:
                    return 0
                # number with `ones` bits; need steps[ones] + 1 <= k? 
                # First op: n -> popcount(n)=ones, then steps[ones] more to reach 1.
                # Total ops = 1 + steps[ones] if ones>1 else (1 if ones==1 and number!=1 else ...)
                # For number > 1: ops = 1 + steps[ones] where ones = popcount(number), and steps[1]=0.
                if ones == 0:
                    return 0
                total_ops = 0 if ones == 1 else 1 + steps[ones]
                # wait: if ones==1, number is power of 2; one op: popcount -> 1, so ops=1 unless number is 1.
                # Number 1 has ones=1 and value 1: 0 ops already 1? Problem usually counts numbers that can be reduced to 1 in at most k operations.
                # For value 1: already 1, 0 operations — typically not < n if n>"1" etc.
                # Standard: while x>1: x=popcount(x); count ops. For x=1, ops=0.
                if ones == 1:
                    # could be 1 or power of 2
                    # powers of 2 need 1 op; we can't distinguish here easily without value.
                    # Actually all numbers with popcount 1 except the number 1 need 1 operation.
                    # Digit DP undercounts distinction for number 1 only — handle started numbers with ones==1: ops=1
                    # Number 1 is the only with value 1; it appears as started with ones=1 and all digits forming 1.
                    total_ops = 1
                else:
                    total_ops = 1 + steps[ones]
                return 1 if total_ops <= k else 0
            lim = int(s[pos]) if tight else 1
            ans = 0
            for bit in range(lim + 1):
                nt = tight and bit == lim
                if not started and bit == 0:
                    ans += dp(pos + 1, nt, 0, 0)
                else:
                    ans += dp(pos + 1, nt, ones + bit, 1)
            return ans % MOD

        # Exclude 0; count numbers in [1, int(s)-1]
        # Our DP counts numbers < s if we force tight start... standard digit DP on s counts <=s if we allow equality.
        # We need < s: run on s with numbers strictly less — classic: dp counts all with tight that are <= pattern when we don't force less.
        # Easier: count <= s-1.
        # Decrement s
        arr = list(s)
        i = len(arr) - 1
        while i >= 0 and arr[i] == "0":
            arr[i] = "1"
            i -= 1
        if i < 0:
            return 0
        arr[i] = "0" if arr[i] == "1" else "0"
        # binary decrement:
        arr = list(s)
        i = len(arr) - 1
        while i >= 0 and arr[i] == "0":
            arr[i] = "1"
            i -= 1
        if i < 0:
            return 0
        arr[i] = "0"
        t = "".join(arr).lstrip("0")
        if not t:
            return 0
        # pad to keep length? Better rewrite DP to count < s directly.
        return self._count_less(s, k)

    def _count_less(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        steps = [0] * (n + 1)
        for i in range(2, n + 1):
            steps[i] = 1 + steps[bin(i).count("1")]

        def ops_for_ones(ones: int) -> int:
            if ones <= 1:
                return 0 if ones == 0 else 1  # treat popcount=1 as 1 op (numbers >=2); number 1 special-cased out by excluding equality of 1 only via < s
            return 1 + steps[ones]

        @lru_cache(None)
        def dfs(pos, tight, ones, started):
            if pos == n:
                if not started:
                    return 0
                return 1 if ops_for_ones(ones) <= k else 0
            up = int(s[pos]) if tight else 1
            res = 0
            for bit in range(0, up + 1):
                nt = tight and (bit == up)
                if not started and bit == 0:
                    res += dfs(pos + 1, nt, 0, 0)
                else:
                    res += dfs(pos + 1, nt, ones + bit, 1)
            return res % MOD

        # dfs on s counts numbers <= s (same length binary with leading zeros as not started)
        # We need < s, so subtract 1 if s itself is k-reducible
        ans = dfs(0, True, 0, 0)
        # check if int(s) is reducible
        ones = s.count("1")
        if ones >= 1 and ops_for_ones(ones) <= k:
            ans = (ans - 1) % MOD
        # number 1: ops=0 <=k; popcount=1; ops_for_ones returns 1 — WRONG for value 1.
        # Fix: if s is "1", numbers < 1: none.
        return ans


def _test():
    s = Solution()
    # Example: s="111", k=1 -> numbers < 7 that are 1-reducible
    # Known example from LC: s = "111", k = 1 → 3
    r = s.countKReducibleNumbers("111", 1)
    assert r == 3, r
    print("LC 3352 OK")


if __name__ == "__main__":
    _test()
