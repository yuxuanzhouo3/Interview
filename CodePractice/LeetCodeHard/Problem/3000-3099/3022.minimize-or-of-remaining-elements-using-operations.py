"""
LC 3022. Minimize OR of Remaining Elements Using Operations — Hard
https://leetcode.com/problems/minimize-or-of-remaining-elements-using-operations/
Best: Bit greed + range-AND sparse table DP — O(30 n log n)
"""

from typing import List


class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        LOG = n.bit_length()
        st = [nums[:]]
        for j in range(1, LOG):
            prev = st[j - 1]
            m = n - (1 << j) + 1
            cur = [0] * m
            half = 1 << (j - 1)
            for i in range(m):
                cur[i] = prev[i] & prev[i + half]
            st.append(cur)

        def range_and(l, r):
            # inclusive
            j = (r - l + 1).bit_length() - 1
            return st[j][l] & st[j][r - (1 << j) + 1]

        def can(target: int) -> bool:
            # max segments with each AND ⊆ target; need >= n-k
            INF = -(10**9)
            dp = [INF] * (n + 1)
            dp[0] = 0
            pref = [0] * (n + 1)  # max dp[0..i]
            for i in range(1, n + 1):
                # find largest j0 in [0,i-1] s.t. AND(j0..i-1) ⊆ target; then all j<=j0 also OK
                # binary search largest j0: AND(j0..i-1) ⊆ target
                lo, hi, j0 = 0, i - 1, -1
                # actually we need the RIGHTMOST j such that AND(j..i-1) ⊆ target
                # AND shrinks as j decreases; once true for j, true for all smaller.
                # So find minimal length = rightmost j via binary search:
                # check mid: AND(mid..i-1) ⊆ target
                lo, hi = 0, i - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if range_and(mid, i - 1) & ~target == 0:
                        j0 = mid  # valid; try more right
                        lo = mid + 1
                    else:
                        hi = mid - 1
                # wait: as mid increases (shorter), AND has more bits, harder to ⊆.
                # Rightmost valid j means shortest valid. Binary search:
                # for mid, if AND(mid..i-1) ⊆ target, try larger mid (shorter).
                # Above finds maximum mid that is valid = rightmost j0. Good.
                if j0 >= 0:
                    # all j in [0,j0] valid? AND(j..i-1) for j<j0 has FEWER bits, still ⊆. Yes.
                    # But j0 from above is max valid mid. Valid j's are 0..j0 only if AND(j0..i-1)⊆ 
                    # and longer ones too. Yes dp[i] = 1 + max(dp[0..j0])
                    # HOWEVER: for j > j0, AND(j..) not ⊆. For j <= j0, longer AND ⊆. Yes.
                    # Wait binary search: if mid valid, we go lo=mid+1 seeking larger. Final j0 = max valid.
                    # Is every mid' < j0 valid? mid' smaller = longer segment = fewer bits = still ⊆ if j0 is. Yes.
                    dp[i] = pref[j0] + 1
                pref[i] = max(pref[i - 1], dp[i])
            return dp[n] >= n - k

        ans = 0
        for b in range(29, -1, -1):
            trial = ans  # without bit b; lower bits free via | ((1<<b)-1)
            allowed = ans | ((1 << b) - 1)
            if not can(allowed):
                ans |= 1 << b
        return ans


def _test():
    s = Solution()
    assert s.minOrAfterOperations([3, 5, 3, 2, 7], 2) == 3
    assert s.minOrAfterOperations([7, 3, 15, 14, 2, 8], 4) == 2
    assert s.minOrAfterOperations([10, 7, 10, 3, 9, 14, 9, 4], 1) == 15
    print("LC 3022 OK")


if __name__ == "__main__":
    _test()
