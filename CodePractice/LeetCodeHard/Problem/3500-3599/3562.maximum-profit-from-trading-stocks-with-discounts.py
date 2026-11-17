"""
LC 3562. Maximum Profit from Trading Stocks with Discounts — Hard
https://leetcode.com/problems/maximum-profit-from-trading-stocks-with-discounts/
Best: 方法一：树形动态规划
"""

class Solution:
    def maxProfit(
        self,
        n: int,
        present: list[int],
        future: list[int],
        hierarchy: list[list[int]],
        budget: int,
    ) -> int:
        max = lambda a, b: a if a > b else b
        g = [[] for _ in range(n + 1)]
        for u, v in hierarchy:
            g[u].append(v)

        def dfs(u: int):
            nxt = [[0, 0] for _ in range(budget + 1)]
            for v in g[u]:
                fv = dfs(v)
                for j in range(budget, -1, -1):
                    for jv in range(j + 1):
                        for pre in (0, 1):
                            val = nxt[j - jv][pre] + fv[jv][pre]
                            if val > nxt[j][pre]:
                                nxt[j][pre] = val

            f = [[0, 0] for _ in range(budget + 1)]
            price = future[u - 1]

            for j in range(budget + 1):
                for pre in (0, 1):
                    cost = present[u - 1] // (pre + 1)
                    if j >= cost:
                        f[j][pre] = max(nxt[j][0], nxt[j - cost][1] + (price - cost))
                    else:
                        f[j][pre] = nxt[j][0]

            return f

        return dfs(1)[budget][0]

def _test():
    s = Solution()
    assert s.maxProfit(n=2, present=[1, 2], future=[4, 3], hierarchy=[[1, 2]], budget=3) == 5
    assert s.maxProfit(n=2, present=[3, 4], future=[5, 8], hierarchy=[[1, 2]], budget=4) == 4
    assert s.maxProfit(n=3, present=[4, 6, 8], future=[7, 9, 11], hierarchy=[[1, 2], [1, 3]], budget=10) == 10
    print("LC 3562 OK")

if __name__ == "__main__":
    _test()
