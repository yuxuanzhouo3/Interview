"""
LC 3985. Palindromic Subarray Sum — Hard
https://leetcode.com/problems/palindromic-subarray-sum/
Best: Manacher on array with separators + prefix sums
"""


class Solution:
    def getSum(self, nums: list[int]) -> int:
        A = [0]
        pre = [0, 0]
        for x in nums:
            A.append(x)
            pre.append(pre[-1] + x)
            A.append(0)
            pre.append(pre[-1])
        n = len(A)
        dp = [0] * n
        l = 0
        r = -1
        for i in range(n):
            dp[i] = max(0, min(r - i, dp[r + l - i] if r + l - i >= 0 else -1))
            while i + dp[i] < n and i - dp[i] >= 0 and A[i - dp[i]] == A[i + dp[i]]:
                dp[i] += 1
            if r < i + dp[i]:
                r = i + dp[i]
                l = i - dp[i]
        res = -10**18
        for i in range(n):
            left = i - dp[i] + 1
            right = i + dp[i]
            res = max(res, pre[right] - pre[left])
        return res


def _test():
    s = Solution()
    assert s.getSum([10, 10]) == 20
    assert s.getSum([1, 2, 3, 2, 1, 5, 6]) == 9
    assert s.getSum([7, 1, 2, 1, 7, 3, 4, 3, 4]) == 18
    assert s.getSum([1, 2, 3, 4, 5]) == 5
    assert s.getSum([1000]) == 1000
    print("LC 3985 OK")


if __name__ == "__main__":
    _test()
