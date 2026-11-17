"""
LC 3953. Maximum Score with Co-Prime Element — Hard
https://leetcode.com/problems/maximum-score-with-co-prime-element/
Best: Mobius inversion to count coprimes; enumerate selected value
"""


class Solution:
    def maxScore(self, nums: list[int], maxVal: int) -> int:
        n = len(nums)
        M = max(maxVal, max(nums))
        freq = [0] * (M + 1)
        for x in nums:
            freq[x] += 1
        mu = [0] * (M + 1)
        lp = [0] * (M + 1)
        primes = []
        mu[1] = 1
        for i in range(2, M + 1):
            if lp[i] == 0:
                lp[i] = i
                primes.append(i)
                mu[i] = -1
            for p in primes:
                v = i * p
                if v > M:
                    break
                lp[v] = p
                if i % p == 0:
                    mu[v] = 0
                    break
                mu[v] = -mu[i]
        multCnt = [0] * (M + 1)
        for d in range(1, M + 1):
            for k in range(d, M + 1, d):
                multCnt[d] += freq[k]
        cop = [0] * (M + 1)
        for d in range(1, M + 1):
            if mu[d] == 0:
                continue
            add = mu[d] * multCnt[d]
            for x in range(d, M + 1, d):
                cop[x] += add
        ans = 0
        for x in range(1, M + 1):
            bad = n - cop[x]
            if freq[x] > 0:
                if x == 1:
                    ans = max(ans, 1)
                else:
                    ans = max(ans, x - bad + 1)
            elif x <= maxVal:
                if bad > 0:
                    ans = max(ans, x - bad)
                else:
                    ans = max(ans, x - 1)
        return ans


def _test():
    s = Solution()
    assert s.maxScore([3, 4, 6], 5) == 4
    assert s.maxScore([1, 2, 3], 4) == 3
    assert s.maxScore([2, 2], 1) == 1
    print("LC 3953 OK")


if __name__ == "__main__":
    _test()
