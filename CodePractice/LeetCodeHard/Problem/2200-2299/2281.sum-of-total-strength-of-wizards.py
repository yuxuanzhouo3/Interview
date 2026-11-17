"""
LC 2281. Sum of Total Strength of Wizards — Hard
https://leetcode.com/problems/sum-of-total-strength-of-wizards/
Best: Monotonic stack + prefix of prefix sums — O(n)
"""


from typing import List


class Solution:
    def totalStrength(self, strength: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(strength)
        left = [-1] * n
        right = [n] * n
        st = []
        for i, x in enumerate(strength):
            while st and strength[st[-1]] >= x:
                st.pop()
            left[i] = st[-1] if st else -1
            st.append(i)
        st = []
        for i in range(n - 1, -1, -1):
            while st and strength[st[-1]] > strength[i]:
                st.pop()
            right[i] = st[-1] if st else n
            st.append(i)
        # pref[i] = sum strength[0..i)
        pref = [0] * (n + 1)
        for i in range(n):
            pref[i + 1] = (pref[i] + strength[i]) % MOD
        # ppref[i] = sum pref[0..i)
        ppref = [0] * (n + 2)
        for i in range(n + 1):
            ppref[i + 1] = (ppref[i] + pref[i]) % MOD

        ans = 0
        for i, x in enumerate(strength):
            L, R = left[i], right[i]
            # subarrays where i is min: l in (L, i], r in [i, R)
            # contribution = x * sum_{l=L+1}^i sum_{r=i}^{R-1} sum(s[l..r])
            # sum(s[l..r]) = pref[r+1]-pref[l]
            left_cnt = i - L
            right_cnt = R - i
            # Using formula with ppref
            sum_right = (ppref[R + 1] - ppref[i + 1] - pref[i] * right_cnt) % MOD
            sum_left = (ppref[i + 1] - ppref[L + 1] - pref[L + 1] * left_cnt) % MOD
            # Actually standard formula:
            # contrib = strength[i] * ( (i-L)*sum_{r=i}^{R-1} pref[r+1] - (R-i)*sum_{l=L+1}^{i} pref[l] )
            sr = (ppref[R + 1] - ppref[i + 1]) % MOD
            sl = (ppref[i + 1] - ppref[L + 1]) % MOD
            contrib = (left_cnt * sr - right_cnt * sl) % MOD
            ans = (ans + x * contrib) % MOD
        return ans


def _test():
    s = Solution()
    assert s.totalStrength([1, 3, 1, 2]) == 44
    assert s.totalStrength([5, 4, 6]) == 213
    print("LC 2281 OK")


if __name__ == "__main__":
    _test()

