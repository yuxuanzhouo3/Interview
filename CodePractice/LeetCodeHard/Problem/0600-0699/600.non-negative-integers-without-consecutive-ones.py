"""
LC 600. Non-negative Integers without Consecutive Ones — Hard
https://leetcode.com/problems/non-negative-integers-without-consecutive-ones/
Best: Fibonacci digit DP on binary — O(log n)
"""


class Solution:
    def findIntegers(self, n: int) -> int:
        # fib[i] = count of valid i-bit numbers (incl leading zeros pattern ending)
        fib = [1, 2]
        for i in range(2, 32):
            fib.append(fib[-1] + fib[-2])

        ans = 0
        prev_bit = 0
        for k in range(31, -1, -1):
            if n & (1 << k):
                ans += fib[k]
                if prev_bit == 1:
                    return ans
                prev_bit = 1
            else:
                prev_bit = 0
        return ans + 1  # include n itself


def _test():
    s = Solution()
    assert s.findIntegers(5) == 5  # 0,1,2,4,5
    assert s.findIntegers(1) == 2
    assert s.findIntegers(2) == 3
    print("LC 600 OK")


if __name__ == "__main__":
    _test()
