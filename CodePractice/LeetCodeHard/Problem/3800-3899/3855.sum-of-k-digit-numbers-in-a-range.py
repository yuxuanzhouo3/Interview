"""
LC 3855. Sum of K-Digit Numbers in a Range — Hard
https://leetcode.com/problems/sum-of-k-digit-numbers-in-a-range/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def sumOfNumbers(self, l: int, r: int, k: int) -> int:
        mod = 10**9 + 7

        n = r - l + 1

        # ((l + r) * (r - l + 1) // 2) % mod
        total = (l + r) * n // 2 % mod

        # pow(r - l + 1, k - 1, mod)
        part1 = pow(n % mod, k - 1, mod)

        # (pow(10, k, mod) - 1)
        part2 = (pow(10, k, mod) - 1) % mod

        # Fermat inverse of 9
        inv9 = pow(9, mod - 2, mod)

        ans = total
        ans = ans * part1 % mod
        ans = ans * part2 % mod
        ans = ans * inv9 % mod

        return ans

def _test():
    s = Solution()
    assert s.sumOfNumbers(1, 2, 2) == 66
    assert s.sumOfNumbers(0, 1, 3) == 444
    assert s.sumOfNumbers(5, 5, 10) == 555555520
    print("LC 3855 OK")


if __name__ == "__main__":
    _test()
