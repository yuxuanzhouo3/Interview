"""
LC 483. Smallest Good Base — Hard
https://leetcode.com/problems/smallest-good-base/
Best: Enumerate length m, binary search base — O(log n * 60)
"""


class Solution:
    def smallestGoodBase(self, n: str) -> str:
        num = int(n)
        # n = (k^m - 1) / (k - 1), m from max bits down to 2
        max_m = num.bit_length()
        for m in range(max_m, 1, -1):
            # find k such that 1 + k + ... + k^(m-1) = num
            lo, hi = 2, int(num ** (1 / (m - 1))) + 1
            while lo <= hi:
                mid = (lo + hi) // 2
                # geometric sum
                s = (mid**m - 1) // (mid - 1)
                if s == num:
                    return str(mid)
                if s < num:
                    lo = mid + 1
                else:
                    hi = mid - 1
        return str(num - 1)


def _test():
    s = Solution()
    assert s.smallestGoodBase("13") == "3"
    assert s.smallestGoodBase("4681") == "8"
    assert s.smallestGoodBase("1000000000000000000") == "999999999999999999"
    print("LC 483 OK")


if __name__ == "__main__":
    _test()
