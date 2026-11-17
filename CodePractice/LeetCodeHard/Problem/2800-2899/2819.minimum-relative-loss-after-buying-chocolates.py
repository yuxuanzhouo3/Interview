"""
LC 2819. Minimum Relative Loss After Buying Chocolates — Hard
https://leetcode.com/problems/minimum-relative-loss-after-buying-chocolates/
Best: Sort + prefix + binary search per query — O((n+q) log n)
"""
from typing import List
from bisect import bisect_right


class Solution:
    def minimumRelativeLosses(self, prices: List[int], queries: List[List[int]]) -> List[int]:
        prices = sorted(prices)
        n = len(prices)
        s = [0] * (n + 1)
        for i, p in enumerate(prices):
            s[i + 1] = s[i] + p

        def calc(k: int, m: int) -> int:
            # loss = p if p<=k else 2k-p
            # take `left` cheapest (<=k) + `right` most expensive
            rbound = bisect_right(prices, k)
            lo, hi = 0, min(m, rbound)
            # binary search optimal left count
            while lo < hi:
                mid = (lo + hi) // 2
                right = m - mid
                # compare marginal: prices[mid] vs 2k - prices[n-right]
                if right == 0 or (mid < n and prices[mid] < 2 * k - prices[n - right]):
                    lo = mid + 1
                else:
                    hi = mid
            left = lo
            # may need to check left and left-1
            def loss(left: int) -> int:
                right = m - left
                if left < 0 or right < 0 or left > rbound or right > n - rbound:
                    return 10**30
                return s[left] + 2 * k * right - (s[n] - s[n - right])

            return min(loss(left), loss(left - 1) if left else 10**30)

        return [calc(k, m) for k, m in queries]


def _test():
    s = Solution()
    assert s.minimumRelativeLosses([1, 9, 22, 10, 19], [[18, 4], [5, 2]]) == [34, -21]
    print("LC 2819 OK")


if __name__ == "__main__":
    _test()
