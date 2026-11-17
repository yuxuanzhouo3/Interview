"""
LC 3864. Minimum Cost to Partition a Binary String — Hard
https://leetcode.com/problems/minimum-cost-to-partition-a-binary-string/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def minCost(self, s: str, encCost: int, flatCost: int) -> int:
        def dfs(l: int, r: int) -> int:
            x = pre[r] - pre[l]
            res = (r - l) * x * encCost if x else flatCost
            if (r - l) % 2 == 0:
                m = (l + r) // 2
                res = min(res, dfs(l, m) + dfs(m, r))
            return res

        n = len(s)
        pre = [0] * (n + 1)
        for i, c in enumerate(s, 1):
            pre[i] = pre[i - 1] + int(c)
        return dfs(0, n)

def _test():
    s = Solution()
    assert s.minCost("1010", 2, 1) == 6
    assert s.minCost("1010", 3, 10) == 12
    assert s.minCost("00", 1, 2) == 2
    print("LC 3864 OK")


if __name__ == "__main__":
    _test()
