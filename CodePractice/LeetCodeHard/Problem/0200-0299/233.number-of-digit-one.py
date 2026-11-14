"""
LC 233. Number of Digit One — Hard
Best: Digit DP / digit place count — O(log n)
"""

class Solution:
    def countDigitOne(self, n: int) -> int:
        if n <= 0:
            return 0
        ans = 0
        place = 1
        while place <= n:
            high = n // (place * 10)
            cur = (n // place) % 10
            low = n % place
            if cur == 0:
                ans += high * place
            elif cur == 1:
                ans += high * place + low + 1
            else:
                ans += (high + 1) * place
            place *= 10
        return ans


def _test():
    s = Solution()
    assert s.countDigitOne(13) == 6
    assert s.countDigitOne(0) == 0
    print("LC 233 OK")


if __name__ == "__main__":
    _test()
