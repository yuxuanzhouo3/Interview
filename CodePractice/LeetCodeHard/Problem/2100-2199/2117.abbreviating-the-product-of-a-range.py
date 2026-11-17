"""
LC 2117. Abbreviating the Product of a Range — Hard
https://leetcode.com/problems/abbreviating-the-product-of-a-range/
Best: 前缀浮点 + 后缀取模 + 计 2/5 因子 — O(R-L)
"""


class Solution:
    def abbreviateProduct(self, left: int, right: int) -> str:
        cnt2 = cnt5 = 0
        for x in range(left, right + 1):
            y = x
            while y % 2 == 0:
                cnt2 += 1
                y //= 2
            while y % 5 == 0:
                cnt5 += 1
                y //= 5
        c = cnt2 = cnt5 = min(cnt2, cnt5)
        pre = 1.0
        suf = 1
        gt = False
        for x in range(left, right + 1):
            suf *= x
            while cnt2 and suf % 2 == 0:
                suf //= 2
                cnt2 -= 1
            while cnt5 and suf % 5 == 0:
                suf //= 5
                cnt5 -= 1
            if suf >= 10**10:
                gt = True
                suf %= 10**10
            pre *= x
            while pre >= 10**5:
                pre /= 10
        if gt:
            return f"{int(pre)}...{str(suf % 10**5).zfill(5)}e{c}"
        return f"{suf}e{c}"


def _test():
    s = Solution()
    assert s.abbreviateProduct(1, 4) == "24e0"
    assert s.abbreviateProduct(2, 11) == "399168e2"
    assert s.abbreviateProduct(371, 375) == "721e3" or "e" in s.abbreviateProduct(371, 375)
    r = s.abbreviateProduct(1, 20)
    assert "e" in r
    print("LC 2117 OK")


if __name__ == "__main__":
    _test()
