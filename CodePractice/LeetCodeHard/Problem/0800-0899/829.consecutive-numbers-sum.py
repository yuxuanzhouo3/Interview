"""
LC 829. Consecutive Numbers Sum — Hard
Best: N = k*(2a+k-1)/2 → odd factors — O(sqrt N)
"""

class Solution:
    def consecutiveNumbersSum(self, n: int) -> int:
        ans = 0
        # n = k*a + k*(k-1)/2 → 2n/k = 2a+k-1
        k = 1
        while k * (k - 1) // 2 < n:
            if (n - k * (k - 1) // 2) % k == 0:
                ans += 1
            k += 1
        return ans


def _test():
    s = Solution()
    assert s.consecutiveNumbersSum(5) == 2
    assert s.consecutiveNumbersSum(9) == 3
    print("LC 829 OK")


if __name__ == "__main__":
    _test()
