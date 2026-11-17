"""
LC 3677. Count Binary Palindromic Numbers — Hard
https://leetcode.com/problems/count-binary-palindromic-numbers/
Best: Enumerate binary palindrome roots — O(log n)
"""


class Solution:
    def countBinaryPalindromes(self, n: int) -> int:
        if n == 0:
            return 1  # 0 itself

        def build(half: int, odd: bool) -> int:
            s = bin(half)[2:]
            if odd:
                return int(s + s[-2::-1], 2) if len(s) > 1 else int(s + s[:-1][::-1], 2)
            return int(s + s[::-1], 2)

        # recount carefully
        def make_pal(half: int, odd: bool) -> int:
            s = bin(half)[2:]
            if odd:
                return int(s + s[-2::-1], 2)
            return int(s + s[::-1], 2)

        ans = 1  # for 0
        # numbers 1..n
        bits = n.bit_length()
        for length in range(1, bits + 1):
            half_len = (length + 1) // 2
            start = 1 << (half_len - 1)
            end = (1 << half_len) - 1
            odd = length % 2 == 1
            for half in range(start, end + 1):
                p = make_pal(half, odd)
                if p > n:
                    break
                if p.bit_length() == length:
                    ans += 1
        return ans


def _test():
    s = Solution()
    assert s.countBinaryPalindromes(9) == 6  # 0,1,2? wait 0,1,3,5,7,9 =6
    assert s.countBinaryPalindromes(0) == 1
    print("LC 3677 OK")


if __name__ == "__main__":
    _test()
