"""
LC 465. Optimal Account Balancing — Hard
https://leetcode.com/problems/optimal-account-balancing/
Best: 净债务 + 子集 DFS / 状态压缩 — O(n·2^n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        bal = defaultdict(int)
        for u, v, amt in transactions:
            bal[u] -= amt
            bal[v] += amt
        debt = [b for b in bal.values() if b != 0]
        n = len(debt)
        if n == 0:
            return 0

        # dp[mask] = 清算 mask 内账户所需最少转账数
        # 若 subset 净和为 0，可用 dp[sub]+dp[mask^sub] 或对连通清算
        # 经典：dfs 从第一个非零债务开始匹配
        def dfs(start: int) -> int:
            while start < n and debt[start] == 0:
                start += 1
            if start == n:
                return 0
            ans = float("inf")
            for i in range(start + 1, n):
                if debt[i] * debt[start] < 0:
                    debt[i] += debt[start]
                    ans = min(ans, 1 + dfs(start + 1))
                    debt[i] -= debt[start]
                    # 剪枝：完全抵消后不必再试更大 i 的同类
                    if debt[i] + debt[start] == 0:
                        break
            return ans

        return dfs(0)


def _test():
    s = Solution()
    assert s.minTransfers([[0, 1, 10], [1, 0, 1], [1, 2, 5], [2, 0, 5]]) == 1
    assert s.minTransfers([[0, 1, 10], [2, 0, 5]]) == 2
    assert s.minTransfers([[0, 1, 1], [1, 2, 1], [2, 0, 1]]) == 0
    assert s.minTransfers([]) == 0
    print("LC 465 OK")


if __name__ == "__main__":
    _test()
