"""
LC 2081. Sum of k-Mirror Numbers — Hard
https://leetcode.com/problems/sum-of-k-mirror-numbers/
Best: Enumerate base-10 palindromes, check base-k — O(limit)
"""


class Solution:
    def kMirror(self, k: int, n: int) -> int:
        def is_pal_base(x: int) -> bool:
            digs = []
            while x:
                digs.append(x % k)
                x //= k
            return digs == digs[::-1]

        def gen():
            # generate decimal palindromes in order
            length = 1
            while True:
                half = (length + 1) // 2
                start = 10 ** (half - 1) if half > 1 else 1
                end = 10 ** half
                for h in range(start, end):
                    s = str(h)
                    if length % 2 == 0:
                        pal = int(s + s[::-1])
                    else:
                        pal = int(s + s[-2::-1])
                    yield pal
                length += 1

        ans = 0
        cnt = 0
        for p in gen():
            if is_pal_base(p):
                ans += p
                cnt += 1
                if cnt == n:
                    return ans


def _test():
    s = Solution()
    assert s.kMirror(2, 5) == 25
    assert s.kMirror(3, 7) == 499
    assert s.kMirror(7, 17) == 20379000
    print("LC 2081 OK")


if __name__ == "__main__":
    _test()

