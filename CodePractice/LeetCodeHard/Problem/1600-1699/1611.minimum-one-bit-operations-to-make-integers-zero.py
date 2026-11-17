"""
LC 1611. Minimum One Bit Operations to Make Integers Zero — Hard
https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/
Best: Gray code reverse — f(n)=2^{msb+1}-1-f(n^(1<<msb)) — O(log n)
"""


class Solution:
    def minimumOneBitOperations(self, n: int) -> int:
        def f(x: int) -> int:
            if x == 0:
                return 0
            msb = x.bit_length() - 1
            return (1 << (msb + 1)) - 1 - f(x ^ (1 << msb))

        return f(n)


def _test():
    s = Solution()
    assert s.minimumOneBitOperations(3) == 2
    assert s.minimumOneBitOperations(6) == 4
    print("LC 1611 OK")


if __name__ == "__main__":
    _test()
