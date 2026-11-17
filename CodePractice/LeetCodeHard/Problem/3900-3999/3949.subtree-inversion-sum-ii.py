"""
LC 3949. Subtree Inversion Sum II — Hard
https://leetcode.com/problems/subtree-inversion-sum-ii/
Best: kamyu104 reference solution (verified on examples)
"""

# Time:  O(n * k)
# Space: O(n + h * k)

# iterative dfs, tree dp
class Solution(object):
    def subtreeInversionSum(self, edges, nums, k):
        """
        :type edges: List[List[int]]
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def iter_dfs():
            result = []
            stk = [(1, (0, -1, result))]
            while stk:
                step, args = stk.pop()
                if step == 1:
                    u, p, ret = args
                    ret[:] = [[nums[u]]*k, [nums[u]]*k]
                    stk.append((4, (u, p, ret)))
                    stk.append((2, (u, p, 0, ret)))
                elif step == 2:
                    u, p, i, ret = args
                    if i == len(adj[u]):
                        continue
                    v = adj[u][i]
                    stk.append((2, (u, p, i+1, ret)))
                    if v == p:
                        continue
                    new_ret = []
                    stk.append((3, (new_ret, ret)))
                    stk.append((1, (v, u, new_ret)))
                elif step == 3:
                    new_ret, ret = args
                    new_dp1, new_dp2 = new_ret
                    dp1, dp2 = ret
                    for i in range(k//2):
                        dp1[i] = max(dp1[i]+new_dp1[(k-2)-i], dp1[(k-2)-i]+new_dp1[i])
                        dp2[i] = min(dp2[i]+new_dp2[(k-2)-i], dp2[(k-2)-i]+new_dp2[i])
                    for i in range(k//2, k):
                        dp1[i] += new_dp1[i]
                        dp2[i] += new_dp2[i]
                    for i in reversed(range(k-1)):
                        dp1[i] = max(dp1[i], dp1[i+1])
                        dp2[i] = min(dp2[i], dp2[i+1])
                elif step == 4:
                    u, p, ret = args
                    dp1, dp2 = ret
                    dp1.insert(0, max(dp1[0], -dp2[-1]))
                    dp2.insert(0, min(dp2[0], -dp1[-1]))
                    dp1.pop()
                    dp2.pop()
            return result[0][0]

        adj = [[] for _ in range(len(nums))]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return iter_dfs()


# Time:  O(n * k)
# Space: O(n + h * k)
# dfs, tree dp

def _test():
    s = Solution()
    assert s.subtreeInversionSum(edges=[[0, 1], [0, 2], [0, 3], [1, 4], [1, 5]], nums=[1, 0, -10, 3, 4, 5], k=2) == 23
    assert s.subtreeInversionSum(edges=[[0, 1], [1, 2]], nums=[5, -10, -10], k=1) == 25
    assert s.subtreeInversionSum(edges=[[0, 1], [0, 2]], nums=[1, -5, -6], k=2) == 12
    assert s.subtreeInversionSum(edges=[[0, 1], [0, 2]], nums=[1, -5, -6], k=3) == 10
    print("LC 3949 OK")


if __name__ == "__main__":
    _test()
