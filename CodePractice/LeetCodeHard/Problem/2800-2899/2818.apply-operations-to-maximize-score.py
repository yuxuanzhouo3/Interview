"""
LC 2818. Apply Operations to Maximize Score — Hard
https://leetcode.com/problems/apply-operations-to-maximize-score/
Best: Prime score + mono stack ranges + greedy pow — O(n log n + U log log U)
"""

from typing import List
import math


class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        mx = max(nums)
        spf = list(range(mx + 1))
        for i in range(2, int(mx**0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, mx + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        def prime_score(x: int) -> int:
            s = set()
            while x > 1:
                p = spf[x]
                s.add(p)
                while x % p == 0:
                    x //= p
            return len(s)

        score = [prime_score(x) for x in nums]
        left = [-1] * n
        right = [n] * n
        st = []
        for i in range(n):
            while st and score[st[-1]] < score[i]:
                st.pop()
            left[i] = st[-1] if st else -1
            st.append(i)
        st = []
        for i in range(n - 1, -1, -1):
            while st and score[st[-1]] <= score[i]:
                st.pop()
            right[i] = st[-1] if st else n
            st.append(i)

        cnt = [(i - left[i]) * (right[i] - i) for i in range(n)]
        order = sorted(range(n), key=lambda i: -nums[i])
        ans = 1
        rem = k
        for i in order:
            if rem <= 0:
                break
            take = min(rem, cnt[i])
            ans = ans * pow(nums[i], take, MOD) % MOD
            rem -= take
        return ans


def _test():
    s = Solution()
    assert s.maximumScore([8, 3, 9, 3, 8], 2) == 81
    assert s.maximumScore([19, 12, 14, 6, 10, 18], 3) == 4788
    print("LC 2818 OK")


if __name__ == "__main__":
    _test()
