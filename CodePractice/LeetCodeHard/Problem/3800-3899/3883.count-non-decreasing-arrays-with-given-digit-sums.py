"""
LC 3883. Count Non Decreasing Arrays With Given Digit Sums — Hard
https://leetcode.com/problems/count-non-decreasing-arrays-with-given-digit-sums/
Best: precompute candidates by digit-sum + monotone DP with prefix sums
"""
MOD = 10**9 + 7


class Solution:
    def countArrays(self, digitSum: list[int]) -> int:
        cands = [[] for _ in range(51)]

        def sod(x: int) -> int:
            s = 0
            while x:
                s += x % 10
                x //= 10
            return s

        for i in range(5001):
            cands[sod(i)].append(i)

        dp = [(-1, 1)]  # (last_value, ways)
        for d in digitSum:
            pref = 0
            idx = 0
            ndp = []
            n = len(dp)
            for val in cands[d]:
                while idx < n and dp[idx][0] <= val:
                    pref = (pref + dp[idx][1]) % MOD
                    idx += 1
                if pref:
                    ndp.append((val, pref))
            dp = ndp
        return sum(v for _, v in dp) % MOD


def _test():
    s = Solution()
    assert s.countArrays([25, 1]) == 6
    assert s.countArrays([1]) == 4
    assert s.countArrays([2, 49, 23]) == 0
    print("LC 3883 OK")


if __name__ == "__main__":
    _test()
