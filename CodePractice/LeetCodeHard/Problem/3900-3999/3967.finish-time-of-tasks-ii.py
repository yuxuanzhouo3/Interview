"""
LC 3967. Finish Time of Tasks II — Hard
https://leetcode.com/problems/finish-time-of-tasks-ii/
Best: kamyu104 reference solution (verified on examples)
"""

# Time:  O(n)
# Space: O(n)

# iterative dfs, tree dp
class Solution(object):
    def finishTime(self, n, edges, baseTime):
        """
        :type n: int
        :type edges: List[List[int]]
        :type baseTime: List[int]
        :rtype: int
        """
        POS_INF, NEG_INF = float("inf"), float("-inf")
        def iter_dfs():
            dp = [0]*n
            stk = [(1, 0, -1)]
            while stk:
                step, u, p = stk.pop()
                if step == 1:
                    stk.append((2, u, p))
                    for v in reversed(adj[u]):
                        if v == p:
                            continue
                        stk.append((1, v, u))
                elif step == 2:
                    mx, mn = NEG_INF, POS_INF
                    for v in adj[u]:
                        if v == p:
                            continue
                        mx, mn = max(mx, dp[v]), min(mn, dp[v])
                    dp[u] = ((2*mx-mn) if mx is not NEG_INF else 0)+baseTime[u]
            return dp
        
        def iter_dfs2():
            def top2(a, b, x, cmp):
                if cmp(x, a):
                    a, b = x, a
                elif cmp(x, b):
                    b = x
                return a, b
    
            result = POS_INF
            stk = [(0, -1, NEG_INF)]
            while stk:
                u, p, t = stk.pop()
                mx1, mx2, mn1, mn2 = NEG_INF, NEG_INF, POS_INF, POS_INF
                for v in adj[u]:
                    x = dp[v] if v != p else t
                    mx1, mx2 = top2(mx1, mx2, x, lambda x, y: x > y)
                    mn1, mn2 = top2(mn1, mn2, x, lambda x, y: x < y)
                result = min(result, ((2*mx1-mn1) if mx1 is not NEG_INF else 0)+baseTime[u])
                for v in reversed(adj[u]):
                    if v == p:
                        continue
                    mx = mx1 if dp[v] != mx1 else mx2
                    mn = mn1 if dp[v] != mn1 else mn2
                    stk.append((v, u, ((2*mx-mn) if mx is not NEG_INF else 0)+baseTime[u]))
            return result
    
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        dp = iter_dfs()
        return iter_dfs2()


# Time:  O(n)
# Space: O(n)
# dfs, tree dp

def _test():
    s = Solution()
    assert s.finishTime(n=3, edges=[[0, 1], [1, 2]], baseTime=[9, 1, 5]) == 14
    assert s.finishTime(n=3, edges=[[0, 1], [0, 2]], baseTime=[4, 7, 6]) == 12
    assert s.finishTime(n=4, edges=[[0, 1], [0, 2], [2, 3]], baseTime=[5, 8, 2, 1]) == 16
    print("LC 3967 OK")


if __name__ == "__main__":
    _test()
