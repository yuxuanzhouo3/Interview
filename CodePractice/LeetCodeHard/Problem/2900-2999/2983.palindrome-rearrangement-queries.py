"""
LC 2983. Palindrome Rearrangement Queries — Hard
https://leetcode.com/problems/palindrome-rearrangement-queries/
Best: Prefix freq + mirror condition — O(n + q)
"""

from typing import List


class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        # prefix freq
        pref = [[0] * 26]
        for ch in s:
            row = pref[-1][:]
            row[ord(ch) - 97] += 1
            pref.append(row)

        def freq(l, r):
            return [pref[r + 1][c] - pref[l][c] for c in range(26)]

        def sub(a, b):
            return [x - y for x, y in zip(a, b)]

        # mismatch prefix of first half vs mirror
        mis = [0] * (half + 1)
        for i in range(half):
            mis[i + 1] = mis[i] + (s[i] != s[n - 1 - i])

        ans = []
        for a, b, c, d in queries:
            # rearrange [a,b] in first half region of s, and [c,d] in second half
            # After rearrange those ranges freely, can whole string be palindrome?
            # Check character budgets for positions that need to match.
            # Standard solution conditions:
            # 1) Outside both rearrange ranges, existing mismatches must be 0
            # 2) The multisets in rearrange ranges must be able to fix the needed mirror pairs

            # Map second half range to mirrored first-half indices
            # Positions i in [0,half) must equal n-1-i
            # We can rearrange chars in [a,b] and [c,d].

            # Check fixed positions outside ranges have no mismatch
            # Fixed in first half: indices not in [a,b]
            # Their mirror may or may not be in [c,d]

            ok = True
            # Use known approach with prefix:
            # For the string to be makeable, considering the two intervals.
            # Simplify via:
            # Let t = s
            # The characters that can be freely reassigned are those in [a,b] U [c,d].
            # Condition from editorial:

            # Check mismatches outside the influence of queries
            def mismatch_outside():
                # positions i where i not in [a,b] and n-1-i not in [c,d] must already match
                # equivalently count
                # Use mis array carefully with intervals
                return True

            # Full correct check:
            # First half index i, mirror j = n-1-i
            # Both fixed => must s[i]==s[j]
            # Collect needed freeness via freqs

            # Pre-check fixed mismatches:
            # Interval of first-half positions with both ends fixed:
            # i notin [a,b] and j=n-1-i notin [c,d]
            # j in [c,d] <=> i in [n-1-d, n-1-c]
            mir_l, mir_r = n - 1 - d, n - 1 - c
            # fixed-fixed positions in [0,half)
            # total mismatches in [0,half) minus those involving rearrange
            # Check each segment of [0,half) that is fixed-fixed
            def range_mis(L, R):
                if L > R:
                    return 0
                return mis[R + 1] - mis[L]

            # positions influenced: i in [a,b] or i in [mir_l, mir_r]
            # fixed-fixed = [0,half) minus those
            # Check three potential fixed segments
            intervals = []
            # subtract [a,b] and [mir_l,mir_r] from [0,half-1]
            marks = [a, b + 1, mir_l, mir_r + 1]
            # sweep
            pts = sorted(set([0, half] + [x for x in marks if 0 <= x <= half]))
            # easier brute for half positions since n<=1e5 — O(n) per query too slow
            # O(1) check:
            # mismatches that must be zero: total mis - mis in union(affected)
            affected_mis = 0
            # union of [a,b] and [mir_l,mir_r] clipped to [0,half)
            segs = []
            for L, R in ((a, b), (mir_l, mir_r)):
                L = max(L, 0)
                R = min(R, half - 1)
                if L <= R:
                    segs.append([L, R])
            segs.sort()
            merged = []
            for L, R in segs:
                if not merged or L > merged[-1][1] + 1:
                    merged.append([L, R])
                else:
                    merged[-1][1] = max(merged[-1][1], R)
            aff = sum(range_mis(L, R) for L, R in merged)
            if mis[half] - aff != 0:
                ok = False

            if ok:
                # frequency condition: for all positions, the multiset of available chars
                # must satisfy mirror constraints.
                # Consider the first half positions that need pairing.
                # Standard freq check:
                # Let F1 = freq[a,b], F2 = freq[c,d]
                # The region of first half that is rearrangeable either directly or via mirror rearrange.
                # Following LC solution:

                # Characters outside rearrange must match their mirrors already checked.
                # For rearrangeable parts, compare freq of s[a..b] with freq of mirror of that = s[n-1-b..n-1-a]
                # similarly for [c,d], and for overlapping influence use inclusion.

                fa = freq(a, b)
                # mirror of [a,b] is [n-1-b, n-1-a]
                fm = freq(n - 1 - b, n - 1 - a)
                fc = freq(c, d)
                # mirror of [c,d] is [n-1-d, n-1-c]
                fcm = freq(n - 1 - d, n - 1 - c)

                # The intersection logic from accepted solutions:
                # We need fa == fm when [c,d] doesn't cover the mirror of [a,b], etc.
                # Casework on how [c,d] relates to mirror([a,b]).

                mb_l, mb_r = n - 1 - b, n - 1 - a  # mirror of [a,b]

                # If [c,d] and mirror([a,b]) disjoint: need freq[a,b]==freq[mirror]
                # If [c,d] covers mirror([a,b]): freer
                # General approach used widely:

                def same(u, v):
                    return u == v

                # Subtract intersection frequencies carefully
                # Simpler O(26) casework from solution:

                # Required: after operations, for every i, we can assign equal chars.
                # Equivalent condition (known):
                # 1) fixed mismatches already handled
                # 2) freq(s[a:b+1]) >= freq of (mirror positions of [a,b] that are NOT in [c,d]) elementwise? 

                # Mirror positions of [a,b]: [mb_l, mb_r]
                # Those not in [c,d] are fixed on the right; their chars must be "provided" by [a,b]
                # Similarly other way.

                # right-fixed part of mirror([a,b]):
                def freq_range_minus(L, R, cutL, cutR):
                    # freq on [L,R] \ [cutL,cutR]
                    if L > R:
                        return [0] * 26
                    total = freq(L, R)
                    iL, iR = max(L, cutL), min(R, cutR)
                    if iL <= iR:
                        inter = freq(iL, iR)
                        return [total[i] - inter[i] for i in range(26)]
                    return total

                # For [a,b] must be able to match the fixed mirrors:
                need_from_left = freq_range_minus(mb_l, mb_r, c, d)
                if any(fa[i] < need_from_left[i] for i in range(26)):
                    ok = False
                # For [c,d] must match fixed mirrors on left (mirror of [c,d] not in [a,b])
                mc_l, mc_r = n - 1 - d, n - 1 - c
                need_from_right = freq_range_minus(mc_l, mc_r, a, b)
                if ok and any(fc[i] < need_from_right[i] for i in range(26)):
                    ok = False

                # Also the "extra" characters in the two ranges must form equal pairs for the remaining flexible positions
                if ok:
                    # Remaining chars in fa after satisfying need_from_left
                    rem_left = [fa[i] - need_from_left[i] for i in range(26)]
                    rem_right = [fc[i] - need_from_right[i] for i in range(26)]
                    # The flexible middle positions (both sides rearrangeable) need rem_left == rem_right as multisets for pairing across?
                    # Actually for fully flexible pairs, leftovers must be equal:
                    if rem_left != rem_right:
                        # Not always — characters can move only within their range, not across ranges.
                        # rem_left is used only for positions whose left is in [a,b] and right in [c,d]
                        # They need rem_left == rem_right as the same multiset to place matching pairs... 
                        # Actually both sides independently choose arrangement; for positions with both free, we need the same counts on both sides for those paired indices.
                        # Number of such paired positions = length of intersection of [a,b] with mirror([c,d]) etc.
                        # rem_left counts chars available for those dual-free pairs (and only those).
                        # So rem_left must equal rem_right.
                        ok = rem_left == rem_right

            ans.append(ok)
        return ans


def _test():
    s = Solution()
    assert s.canMakePalindromeQueries("abcabc", [[1, 1, 3, 5], [0, 2, 5, 5]]) == [True, True]
    assert s.canMakePalindromeQueries("abbcdecbba", [[0, 2, 7, 9]]) == [False]
    assert s.canMakePalindromeQueries("acbcab", [[1, 2, 4, 5]]) == [True]
    print("LC 2983 OK")


if __name__ == "__main__":
    _test()
