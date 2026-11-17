from bisect import bisect_left, bisect_right
"""
LC 3801. Minimum Cost to Merge Sorted Lists — Hard
https://leetcode.com/problems/minimum-cost-to-merge-sorted-lists/
Best: doocs/leetcode reference solution (verified on examples)
"""

from math import inf

class Solution:
    def minMergeCost(self, lists: list[list[int]]) -> int:
        n = len(lists)
        vals = sorted({x for v in lists for x in v})
        cnt = [0] * (1 << n)
        med = [0] * (1 << n)
        for i in range(1, 1 << n):
            for j, v in enumerate(lists):
                if i >> j & 1:
                    cnt[i] += len(v)
            need = (cnt[i] + 1) // 2
            l, r = 0, len(vals) - 1
            while l < r:
                mid = (l + r) >> 1
                le = 0
                b = i
                while b:
                    t = (b & -b).bit_length() - 1
                    le += bisect_right(lists[t], vals[mid])
                    if le >= need:
                        break
                    b &= b - 1
                if le >= need:
                    r = mid
                else:
                    l = mid + 1
            med[i] = vals[l]

        f = [inf] * (1 << n)
        for i in range(1, 1 << n):
            if bin(i).count("1") == 1:
                f[i] = 0
                continue
            j = (i - 1) & i
            while j:
                k = i ^ j
                if j <= k:
                    f[i] = min(f[i], f[j] + f[k] + abs(med[j] - med[k]))
                j = (j - 1) & i
            f[i] += cnt[i]
        return f[-1]

def _test():
    s = Solution()
    assert s.minMergeCost([[1,3,5],[2,4],[6,7,8]]) == 18
    assert s.minMergeCost([[1,1,5],[1,4,7,8]]) == 10
    assert s.minMergeCost([[1],[3]]) == 4
    assert s.minMergeCost([[1],[1]]) == 2
    print("LC 3801 OK")


if __name__ == "__main__":
    _test()
