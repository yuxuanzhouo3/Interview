"""
LC 3463. Check If Digits Are Equal in String After Operations II — Hard
https://leetcode.com/problems/check-if-digits-are-equal-in-string-after-operations-ii/
Best: Combinatorial (mod 2 & 5) / Lucas — O(n)
"""


class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # repeatedly replace with (s[i]+s[i+1])%10 until length 2; check equal
        # final two digits are weighted sums with binomial coeffs C(n-2, i)
        n = len(s)
        digits = [ord(c) - 48 for c in s]

        def comb_mod(n_, k_, p):
            if k_ < 0 or k_ > n_:
                return 0
            # compute C(n_,k_) % p for small p via Lucas
            def lucas(n, k, p):
                ans = 1
                while n or k:
                    ni, ki = n % p, k % p
                    if ki > ni:
                        return 0
                    # C(ni, ki) % p
                    num = den = 1
                    for i in range(ki):
                        num = num * (ni - i) % p
                        den = den * (i + 1) % p
                    ans = ans * num * pow(den, p - 2, p) % p
                    n //= p
                    k //= p
                return ans

            return lucas(n_, k_, p)

        # need final digit0 == digit1, each is sum C(n-2,i)*d[i] and C(n-2,i)*d[i+1] mod 10
        # compare mod 2 and mod 5 via CRT
        def eval_mod(p):
            a = b = 0
            for i in range(n - 1):
                c = comb_mod(n - 2, i, p)
                a = (a + c * digits[i]) % p
                b = (b + c * digits[i + 1]) % p
            return a == b

        # Actually final length-2: left = sum C(n-2,i)*s[i], right = sum C(n-2,i)*s[i+1]
        # We need left ≡ right (mod 10). Check mod 2 and mod 5.
        return eval_mod(2) and eval_mod(5)


def _test():
    s = Solution()
    assert s.hasSameDigits("3902") is True
    assert s.hasSameDigits("34789") is False
    print("LC 3463 OK")


if __name__ == "__main__":
    _test()
