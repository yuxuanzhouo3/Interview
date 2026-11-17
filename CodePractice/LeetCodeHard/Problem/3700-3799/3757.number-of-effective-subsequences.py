"""
LC 3757. Number of Effective Subsequences — Hard
https://leetcode.com/problems/number-of-effective-subsequences/
Best: Complement: subsequences that preserve OR — O(n * 32)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Set
import math
import heapq
import bisect
from collections import defaultdict, Counter, deque
from functools import lru_cache
from math import gcd, isqrt


class Solution:
    def countEffective(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        full = 0
        for x in nums:
            full |= x
        # total non-empty subsequences
        total = (pow(2, n, MOD) - 1) % MOD
        # count subsequences whose removal does NOT decrease OR
        # i.e. remaining OR == full, meaning the removed subseq doesn't uniquely cover any bit of full
        # Equivalent: for every bit in full, at least one number providing it remains
        # = total non-empty - those that decrease OR
        # Count subsequences S (removed) such that OR(all \ S) < full
        # Easier: count non-empty S that are effective via complement of "remaining still has full OR"
        # Remaining is also a subsequence (possibly empty). Empty remaining OR=0 < full (if full>0).
        # Number of subsequences (as remaining) with OR == full, then removed = complement, including empty removed.
        # Let R = number of subsequences (incl empty) with OR == full.
        # Then non-effective removals: those where remaining OR == full, i.e. R choices of remaining, removed = complement.
        # But removed must be non-empty subsequence => remaining != all elements.
        # Total removals = 2^n - 1.
        # Effective = total - (R - [OR(all)==full]) where R-1 if full OR equals full (always), subtracting the empty removal.
        # Non-effective non-empty removals = number of proper remaining with OR==full = R - 1 (exclude remaining=all? remaining=all means removed empty).
        # R includes empty remaining? Empty OR=0, included only if full==0.
        # Standard approach: for each bit, track providers.

        # Count subsequences whose OR equals `full` using SOS / digit DP on bits:
        # Inclusion: number of subsequences with OR | full == full and not missing bits.
        # Use: count subsequences that avoid at least one required bit's all providers.

        # Let bits of full be B. Effective means exists bit b in B such that all providers of b are inside removed S.
        # Inclusion-exclusion over set of bits that are fully covered by S.

        bit_providers = []
        for b in range(32):
            if (full >> b) & 1:
                idxs = [i for i, x in enumerate(nums) if (x >> b) & 1]
                bit_providers.append(idxs)

        m = len(bit_providers)
        # IE over non-empty subsets of bits that are wiped
        bad = 0  # actually we want effective count directly via IE
        # |union A_b| where A_b = subsequences containing all providers of b (and anything else)
        # A_b: must include all providers of b; other n-|P_b| elements free: 2^{n-|P|} 
        # but subsequence non-empty automatic if |P|>=1
        from itertools import combinations
        providers_sets = [set(p) for p in bit_providers]
        effective = 0
        for r in range(1, m + 1):
            for comb in combinations(range(m), r):
                union = set()
                for j in comb:
                    union |= providers_sets[j]
                ways = pow(2, n - len(union), MOD)
                if r & 1:
                    effective = (effective + ways) % MOD
                else:
                    effective = (effective - ways) % MOD
        return effective % MOD


def _test():

    s = Solution()
    assert s.countEffective([1, 2, 3]) == 3
    assert s.countEffective([7, 4, 6]) == 4

    print("LC 3757 OK")


if __name__ == "__main__":
    _test()
