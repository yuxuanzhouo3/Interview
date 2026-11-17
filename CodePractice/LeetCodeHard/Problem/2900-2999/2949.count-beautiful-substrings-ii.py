"""
LC 2949. Count Beautiful Substrings II — Hard
https://leetcode.com/problems/count-beautiful-substrings-ii/
Best: Balance prefix + modulo period — O(n * sqrt(k))
"""


class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set("aeiou")
        # beautiful: vowels==consonants=v, and v*v % k == 0 => v^2 ≡ 0 (mod k)
        # length 2v, need v % root == 0 where root is minimal m with m^2 % k == 0? 
        # Actually need (v*v) % k == 0. Let period = min positive p s.t. p^2 % k == 0? No:
        # v^2 ≡ 0 mod k means k | v^2. Let g be square-free part handling:
        # Find smallest d such that d*d % k == 0... wait we need k | v^2, not that.
        # Equivalent: for every prime p^e || k, need p^{ceil(e/2)} | v.
        # So let m = product p^{ceil(e/2)}; need m | v.
        m = 1
        x = k
        p = 2
        while p * p <= x:
            if x % p == 0:
                e = 0
                while x % p == 0:
                    x //= p
                    e += 1
                need = (e + 1) // 2
                m *= p ** need
            p += 1
        if x > 1:
            m *= x  # e=1, ceil=1
        # track balance: vowel +1, consonant -1; at equal, vowels = len/2
        # when balance returns to same value with len/2 % m == 0
        # Use: pref of (vowels - consonants). Start 0.
        # Also need vowels count % m == 0. Since vowels = (len)/2 and consonants=len/2 when balance 0.
        # len = #v+#c, balance = #v-#c. When balance=0, #v=#c=len/2.
        # Store for each balance the list of vowel-counts (or positions) at which we saw it.
        # Optimized: store count of times we saw (balance, vowels % m)
        from collections import defaultdict
        seen = defaultdict(int)
        seen[(0, 0)] = 1
        bal = 0
        vows = 0
        ans = 0
        for ch in s:
            if ch in vowels:
                bal += 1
                vows += 1
            else:
                bal -= 1
            key = (bal, vows % m)
            ans += seen[key]
            seen[key] += 1
        return ans


def _test():
    s = Solution()
    assert s.beautifulSubstrings("baeyh", 2) == 2
    assert s.beautifulSubstrings("abba", 1) == 3
    assert s.beautifulSubstrings("bcdf", 1) == 0
    print("LC 2949 OK")


if __name__ == "__main__":
    _test()
