"""
LC 3962. Maximum Subarray Sum After at Most K Swaps — Hard
https://leetcode.com/problems/maximum-subarray-sum-after-at-most-k-swaps/
Best: enumerate subarrays + Fenwick for k best outside/inside swaps
"""


class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.count = [0] * (n + 1)
        self.sum = [0] * (n + 1)

    def add(self, idx: int, cnt: int, val: int):
        idx += 1
        while idx <= self.n:
            self.count[idx] += cnt
            self.sum[idx] += val
            idx += idx & -idx

    def prefix_count(self, idx: int) -> int:
        res = 0
        while idx > 0:
            res += self.count[idx]
            idx -= idx & -idx
        return res

    def prefix_sum(self, idx: int) -> int:
        res = 0
        while idx > 0:
            res += self.sum[idx]
            idx -= idx & -idx
        return res

    def kth(self, k: int) -> int:
        idx = 0
        bit = 1 << (self.n.bit_length())
        while bit:
            nxt = idx + bit
            if nxt <= self.n and self.count[nxt] < k:
                idx = nxt
                k -= self.count[nxt]
            bit >>= 1
        return idx

    def sum_smallest(self, k: int, values: list[int]) -> int:
        if k <= 0:
            return 0
        pos = self.kth(k)
        before = self.prefix_count(pos)
        return self.prefix_sum(pos) + (k - before) * values[pos]

    def sum_largest(self, k: int, values: list[int]) -> int:
        total = self.prefix_count(self.n)
        if k <= 0:
            return 0
        if k >= total:
            return self.prefix_sum(self.n)
        return self.prefix_sum(self.n) - self.sum_smallest(total - k, values)


class Solution:
    def maxSum(self, nums: list[int], k: int) -> int:
        n = len(nums)
        values = sorted(set(nums))
        m = len(values)
        rank = {v: i for i, v in enumerate(values)}
        idx = [rank[x] for x in nums]
        ans = -(10**18)
        for l in range(n):
            inside = Fenwick(m)
            outside = Fenwick(m)
            for i in range(n):
                outside.add(idx[i], 1, nums[i])
            window = 0
            for r in range(l, n):
                outside.add(idx[r], -1, -nums[r])
                inside.add(idx[r], 1, nums[r])
                window += nums[r]
                in_cnt = r - l + 1
                out_cnt = n - in_cnt
                limit = min(k, in_cnt, out_cnt)
                if limit == 0:
                    ans = max(ans, window)
                    continue
                lo, hi, best = 1, limit, 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    small = inside.kth(mid)
                    large = outside.kth(out_cnt - mid + 1)
                    if values[small] < values[large]:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best == 0:
                    ans = max(ans, window)
                else:
                    cand = window + outside.sum_largest(best, values) - inside.sum_smallest(best, values)
                    ans = max(ans, cand)
        return ans


def _test():
    s = Solution()
    assert s.maxSum([1, -1, 0, 2], 1) == 3
    assert s.maxSum([4, 3, 2, 4], 2) == 13
    assert s.maxSum([-1, -2], 0) == -1
    print("LC 3962 OK")


if __name__ == "__main__":
    _test()
