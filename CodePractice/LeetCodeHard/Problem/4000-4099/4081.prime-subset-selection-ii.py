"""
LC 4081. Prime Subset Selection II — Hard
https://leetcode.com/problems/prime-subset-selection-ii/
Best: 0-1 knapsack storing best subset tuple for each sum
"""


class Solution:
    def maxPrimeSubset(self, n: int, s: int) -> list[int]:
        sieve = [True] * (n + 1)
        primes = []
        for i in range(2, n + 1):
            if sieve[i]:
                primes.append(i)
                step = i
                start = i * i
                for j in range(start, n + 1, step):
                    sieve[j] = False

        # dp[cap] = best subset (as sorted tuple) achieving exact sum cap
        # compare: shorter better, then lex smaller
        dp: list[tuple[int, ...] | None] = [None] * (s + 1)
        dp[0] = ()
        for p in primes:
            for cap in range(s, p - 1, -1):
                if dp[cap - p] is None:
                    continue
                cand = tuple(sorted(dp[cap - p] + (p,)))
                if dp[cap] is None:
                    dp[cap] = cand
                else:
                    old = dp[cap]
                    if len(cand) < len(old) or (len(cand) == len(old) and cand < old):
                        dp[cap] = cand

        best = None
        for cap in range(s, 0, -1):
            if dp[cap] is not None:
                if best is None:
                    best = dp[cap]
                else:
                    # same sum? we iterate high to low so first is max sum
                    break
        # Among max sum already taken as highest cap with any subset.
        # But multiple caps: we want MAX sum first — so scan from s down.
        best = None
        best_sum = -1
        for cap in range(1, s + 1):
            if dp[cap] is None:
                continue
            if cap > best_sum:
                best_sum = cap
                best = dp[cap]
            elif cap == best_sum:
                if len(dp[cap]) < len(best) or (len(dp[cap]) == len(best) and dp[cap] < best):
                    best = dp[cap]
        return list(best) if best else []


def _test():
    s = Solution()
    assert s.maxPrimeSubset(7, 20) == [2, 3, 5, 7]
    assert s.maxPrimeSubset(15, 18) == [5, 13]
    assert s.maxPrimeSubset(5, 1) == []
    print("LC 4081 OK")


if __name__ == "__main__":
    _test()
